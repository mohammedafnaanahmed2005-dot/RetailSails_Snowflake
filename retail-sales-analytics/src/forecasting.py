"""Monthly forecasting of net revenue; contribution derived from drivers.

Methods (all evaluated on the same holdout = last `holdout` months):
  * recent_mean   - average of the last 3 months (simple baseline)
  * linear_trend  - OLS straight line on month index, with 95% prediction interval
  * snowflake_ml  - SNOWFLAKE.ML.FORECAST (gradient-boosted, Snowflake-managed),
                    used only if the role can create forecast models.

Contribution per forecast month is rebuilt with the driver model in
src/scenarios.py from the trailing-period unit economics, so revenue and
contribution stay consistent and scenarios change underlying drivers.
"""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd

from src.scenarios import SCENARIO_PRESETS, Adjustments, Drivers, compute

MIN_HISTORY_MONTHS = 6
HORIZONS = (3, 6, 12)


def validate_history(hist: pd.DataFrame) -> list[str]:
    """Return blocking problems with the input series (empty list = OK)."""
    problems = []
    if hist is None or len(hist) == 0:
        return ["No monthly history available for the selected filters."]
    if len(hist) < MIN_HISTORY_MONTHS:
        problems.append(f"Only {len(hist)} months of history; at least {MIN_HISTORY_MONTHS} are required to forecast.")
    months = pd.to_datetime(hist["SALE_MONTH"]).sort_values()
    expected = pd.date_range(months.iloc[0], months.iloc[-1], freq="MS")
    if len(expected) != len(months):
        problems.append(f"{len(expected) - len(months)} month(s) missing in the history (gaps).")
    if (hist["NET_REVENUE_INR"].fillna(0) <= 0).all():
        problems.append("Net revenue is zero for all months.")
    return problems


def _future_months(last: pd.Timestamp, h: int) -> list[pd.Timestamp]:
    return list(pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS"))


def recent_mean(y: np.ndarray, h: int, window: int = 3) -> pd.DataFrame:
    m = float(np.mean(y[-window:]))
    sd = float(np.std(y[-window:], ddof=1)) if len(y[-window:]) > 1 else 0.0
    return pd.DataFrame({"forecast": [m] * h, "lower": [m - 1.96 * sd] * h, "upper": [m + 1.96 * sd] * h})


def linear_trend(y: np.ndarray, h: int) -> pd.DataFrame:
    n = len(y)
    x = np.arange(n, dtype=float)
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    s = math.sqrt(float(np.sum(resid ** 2)) / max(n - 2, 1))
    xf = np.arange(n, n + h, dtype=float)
    se = s * np.sqrt(1 + 1 / n + (xf - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    f = b0 + b1 * xf
    return pd.DataFrame({"forecast": f, "lower": f - 1.96 * se, "upper": f + 1.96 * se})


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    nz = actual != 0
    return {
        "MAE_INR": float(np.mean(np.abs(err))),
        "MAPE_PCT": float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else None,
        "BIAS_INR": float(np.mean(err)),
    }


def _snowflake_ml(session, months: list, y: np.ndarray, h: int) -> pd.DataFrame | None:
    """Train a temporary SNOWFLAKE.ML.FORECAST model; returns None if unavailable."""
    suffix = uuid.uuid4().hex[:10].upper()
    tbl, model = f"TMP_FC_IN_{suffix}", f"TMP_FC_MODEL_{suffix}"
    try:
        df = pd.DataFrame({"TS": pd.to_datetime(months), "Y": y.astype(float)})
        session.write_pandas(df, tbl, auto_create_table=True, table_type="temporary", overwrite=True,
                             quote_identifiers=False, use_logical_type=True)
        session.sql(f"""CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {model}(
            INPUT_DATA => TABLE(SELECT TS::TIMESTAMP_NTZ AS TS, Y FROM {tbl}),
            TIMESTAMP_COLNAME => 'TS', TARGET_COLNAME => 'Y')""").collect()
        res = session.sql(f"CALL {model}!FORECAST(FORECASTING_PERIODS => {int(h)})").to_pandas()
        res.columns = [c.upper() for c in res.columns]
        return pd.DataFrame({"forecast": res["FORECAST"].astype(float).values,
                             "lower": res["LOWER_BOUND"].astype(float).values,
                             "upper": res["UPPER_BOUND"].astype(float).values})
    except Exception:
        return None
    finally:
        for stmt in (f"DROP SNOWFLAKE.ML.FORECAST IF EXISTS {model}", f"DROP TABLE IF EXISTS {tbl}"):
            try:
                session.sql(stmt).collect()
            except Exception:
                pass


METHODS = {"recent_mean": "Recent 3-month mean (baseline)", "linear_trend": "Linear trend",
           "snowflake_ml": "Snowflake ML FORECAST"}


def _predict(method, session, months, y, h):
    if method == "recent_mean":
        return recent_mean(y, h)
    if method == "linear_trend":
        return linear_trend(y, h)
    if method == "snowflake_ml" and session is not None:
        return _snowflake_ml(session, months, y, h)
    return None


def evaluate(hist: pd.DataFrame, session=None, holdout: int = 3, use_ml: bool = True) -> pd.DataFrame:
    """Backtest each method on the last `holdout` months."""
    hist = hist.sort_values("SALE_MONTH")
    y = hist["NET_REVENUE_INR"].astype(float).values
    months = list(pd.to_datetime(hist["SALE_MONTH"]))
    if len(y) - holdout < MIN_HISTORY_MONTHS - 1:
        return pd.DataFrame()
    rows = []
    for m in METHODS:
        if m == "snowflake_ml" and not use_ml:
            continue
        p = _predict(m, session, months[:-holdout], y[:-holdout], holdout)
        if p is None:
            continue
        actual = y[-holdout:]
        cover = float(np.mean((actual >= p["lower"].values) & (actual <= p["upper"].values)) * 100)
        rows.append({"method": m, "label": METHODS[m], **metrics(actual, p["forecast"].values),
                     "INTERVAL_COVERAGE_PCT": cover, "holdout_months": holdout})
    return pd.DataFrame(rows).sort_values("MAE_INR") if rows else pd.DataFrame()


def forecast(hist: pd.DataFrame, horizon: int, method: str, session=None,
             scenarios: dict[str, Adjustments] | None = None, driver_window: int = 6) -> dict:
    """Return forecast table with revenue + driver-derived contribution per scenario."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    problems = validate_history(hist)
    if problems:
        raise ValueError(" ".join(problems))
    hist = hist.sort_values("SALE_MONTH").reset_index(drop=True)
    y = hist["NET_REVENUE_INR"].astype(float).values
    months = list(pd.to_datetime(hist["SALE_MONTH"]))
    pred = _predict(method, session, months, y, horizon)
    used = method
    if pred is None:  # e.g. ML not authorised -> transparent fallback
        pred, used = linear_trend(y, horizon), "linear_trend"
    pred["forecast"] = pred["forecast"].clip(lower=0)
    pred["lower"] = pred["lower"].clip(lower=0)

    # Unit economics from the trailing window (sum of totals -> driver ratios)
    tail = hist.tail(driver_window)
    totals = {c: float(tail[c].sum()) for c in ["GROSS_SALES_INR", "UNITS", "TXN_COUNT", "DISCOUNT_INR",
                                                "RETURNS_INR", "COGS_INR", "SHIPPING_INR", "MARKETING_INR",
                                                "NET_REVENUE_INR"] if c in tail}
    totals["REVENUE_INR"] = totals["GROSS_SALES_INR"] - totals["DISCOUNT_INR"]
    d_window = Drivers.from_totals(totals)
    net_per_unit = compute(d_window)["net_revenue"] / d_window.units if d_window.units else 0

    scenarios = scenarios or SCENARIO_PRESETS
    rows = []
    for i, ts in enumerate(_future_months(months[-1], horizon)):
        base_net = float(pred["forecast"].iloc[i])
        scale = base_net / net_per_unit / d_window.units if net_per_unit and d_window.units else 0
        d_month = Drivers(units=d_window.units * scale, txns=d_window.txns * scale,
                          list_price=d_window.list_price, discount_rate=d_window.discount_rate,
                          return_rate=d_window.return_rate, cogs_per_unit=d_window.cogs_per_unit,
                          shipping_per_txn=d_window.shipping_per_txn, marketing_per_txn=d_window.marketing_per_txn)
        row = {"SALE_MONTH": ts.date(), "FORECAST_NET_REVENUE_INR": base_net,
               "LOWER_95_INR": float(pred["lower"].iloc[i]), "UPPER_95_INR": float(pred["upper"].iloc[i])}
        for name, adj in scenarios.items():
            r = compute(d_month, adj)
            key = name.upper().replace(" ", "_")
            row[f"{key}_NET_REVENUE_INR"] = r["net_revenue"]
            row[f"{key}_CONTRIBUTION_INR"] = r["contribution"]
        rows.append(row)
    out = pd.DataFrame(rows)
    return {"table": out, "method_used": used, "drivers": d_window, "driver_window": driver_window,
            "history_months": len(hist), "fallback": used != method}
