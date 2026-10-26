import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ai_assistant as AI  # noqa: E402
from src import analytics as A  # noqa: E402
from src import data_loader as L  # noqa: E402
from src import forecasting as F  # noqa: E402
from src import report_generator as RG  # noqa: E402
from src import scenarios as S  # noqa: E402
from src import validation as V  # noqa: E402

SAMPLE = ROOT / "data" / "retail_sales_analytics_sample.csv"


def _sample_bytes():
    return SAMPLE.read_bytes()


def _validate(raw: pd.DataFrame, existing=None):
    mp = V.suggest_mapping(list(raw.columns))
    typed, notes = V.coerce_and_derive(V.apply_mapping(raw, mp), mp)
    return V.validate(typed, mp, existing), mp, notes


# ------------------------------------------------------------- CSV validation
def test_sample_schema_all_columns_mapped():
    raw = V.read_csv(_sample_bytes())
    assert len(raw) == 1200
    mp = V.suggest_mapping(list(raw.columns))
    assert all(mp[c] == c for c in V.REQUIRED_COLUMNS)


def test_sample_is_fully_valid_and_keeps_literal_none():
    raw = V.read_csv(_sample_bytes())
    assert (raw.promotion_type == "None").sum() == 316  # not turned into NaN
    res, _, _ = _validate(raw)
    assert res["summary"]["valid_rows"] == 1200
    assert res["summary"]["rows_with_warnings"] == 0


def _mini():
    raw = V.read_csv(_sample_bytes()).head(6).copy()
    return raw


def test_missing_and_duplicate_ids_flagged_not_dropped():
    raw = _mini()
    raw.loc[1, "transaction_id"] = raw.loc[0, "transaction_id"]
    raw.loc[2, "transaction_id"] = np.nan
    res, _, _ = _validate(raw)
    fr = res["frame"]
    assert len(fr) == 6  # nothing dropped
    assert "duplicate transaction_id" in fr.loc[0, "_errors"] and "duplicate transaction_id" in fr.loc[1, "_errors"]
    assert "missing transaction_id" in fr.loc[2, "_errors"]
    assert res["summary"]["duplicate_ids_in_file"] == 2


def test_invalid_dates_numbers_quantity_discount():
    raw = _mini()
    raw.loc[0, "transaction_date"] = "2025-13-45"
    raw.loc[1, "quantity"] = "-2"
    raw.loc[2, "discount_pct"] = "150"
    raw.loc[3, "unit_price_inr"] = "abc"
    res, _, _ = _validate(raw)
    e = res["frame"]["_errors"]
    assert "invalid transaction_date" in e[0]
    assert "quantity must be > 0" in e[1]
    assert "discount_pct outside 0-100" in e[2]
    assert "unit_price_inr not numeric" in e[3]
    assert res["summary"]["valid_rows"] == 2


def test_existing_ids_warn_on_append():
    raw = _mini()
    res, _, _ = _validate(raw, existing={raw.transaction_id[0]})
    assert "already exists" in res["frame"]["_warnings"][0]
    assert res["frame"]["_valid"][0]  # warning, not error


def test_column_mapping_with_different_headers_and_derivation():
    raw = _mini()[["transaction_id", "transaction_date", "product_name", "category", "region", "sales_channel",
                   "customer_segment", "quantity", "unit_price_inr", "discount_pct", "net_revenue_inr",
                   "cost_of_goods_inr", "shipping_cost_inr", "marketing_cost_inr"]]
    raw = raw.rename(columns={"transaction_id": "Order ID", "quantity": "Qty", "unit_price_inr": "Unit Price",
                              "transaction_date": "Order Date", "cost_of_goods_inr": "COGS"})
    res, mp, notes = _validate(raw)
    assert mp["transaction_id"] == "Order ID" and mp["quantity"] == "Qty" and mp["cost_of_goods_inr"] == "COGS"
    assert any("gross_revenue_inr derived" in n for n in notes)
    assert res["summary"]["valid_rows"] == 6
    sample = V.read_csv(_sample_bytes()).head(6)
    assert np.allclose(res["frame"]["revenue_inr"].values, sample.revenue_inr.astype(float).values, atol=0.02)


def test_missing_critical_column_blocks():
    raw = _mini().drop(columns=["net_revenue_inr"])
    res, _, _ = _validate(raw)
    assert res["blocking"] and "net_revenue_inr" in res["summary"]["unmapped_critical_columns"]


def test_missing_costs_marked_unavailable_not_invented():
    raw = _mini().drop(columns=["cost_of_goods_inr", "net_profit_inr"])
    res, _, notes = _validate(raw)
    assert any("UNAVAILABLE" in n for n in notes)
    assert res["frame"]["cost_of_goods_inr"].isna().all()
    assert res["frame"]["net_profit_inr"].isna().all()


def test_clean_for_load_types():
    res, _, _ = _validate(_mini())
    out = V.clean_for_load(res["frame"])
    assert list(out.columns) == L.TARGET_COLS
    assert str(out.RETURNED.dtype) == "boolean"


def test_loader_name_guards():
    assert L.safe_table_name("sales_x") == "SALES_X"
    for bad in ["x; drop table y", "1abc", "", "a-b"]:
        with pytest.raises(ValueError):
            L.safe_table_name(bad)
    assert L.safe_file_name("../../etc/pass wd.csv") == ".._.._etc_pass_wd.csv"


# ------------------------------------------------------------- calculations
def _pandas_kpis(df):
    df = df.copy()
    for c in V.NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    con = df.net_revenue_inr - df.cost_of_goods_inr - df.shipping_cost_inr - df.marketing_cost_inr
    n = df.transaction_id.nunique()
    return {"rev": df.net_revenue_inr.sum(), "con": con.sum(), "margin": con.sum() / df.net_revenue_inr.sum() * 100,
            "aov": df.net_revenue_inr.sum() / n, "rr": (df.returned == "True").sum() / n * 100, "n": n,
            "gross": (df.unit_price_inr * df.quantity).sum(), "disc": df.discount_amount_inr.sum()}


def test_contribution_reconciles_with_recorded_profit():
    df = V.read_csv(_sample_bytes())
    k = _pandas_kpis(df)
    assert round(k["con"], 2) == round(df.net_profit_inr.astype(float).sum(), 2) == 1648517.53
    assert round(k["rev"], 2) == 9659156.37
    assert round(k["margin"], 2) == 17.07 and round(k["aov"], 2) == 8049.30 and round(k["rr"], 2) == 5.67


def test_returns_and_discounts_not_double_counted():
    df = V.read_csv(_sample_bytes())
    for c in V.NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    # revenue already net of discount; net revenue already net of returns
    assert np.allclose(df.gross_revenue_inr - df.discount_amount_inr, df.revenue_inr, atol=0.02)
    assert np.allclose(df.revenue_inr - df.return_amount_inr, df.net_revenue_inr, atol=0.02)
    ret = df[df.returned_ == True] if "returned_" in df else df[df.returned == "True"]  # noqa: E712
    assert (ret.net_revenue_inr == 0).all()  # full reversal in the sample


def test_scenario_baseline_reproduces_totals_and_zero_revenue_margin():
    df = V.read_csv(_sample_bytes())
    k = _pandas_kpis(df)
    for c in V.NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    tot = {"GROSS_SALES_INR": k["gross"], "UNITS": df.quantity.sum(), "TXN_COUNT": k["n"], "DISCOUNT_INR": k["disc"],
           "REVENUE_INR": df.revenue_inr.sum(), "RETURNS_INR": df.return_amount_inr.sum(), "COGS_INR": df.cost_of_goods_inr.sum(),
           "SHIPPING_INR": df.shipping_cost_inr.sum(), "MARKETING_INR": df.marketing_cost_inr.sum()}
    base = S.compute(S.Drivers.from_totals(tot))
    assert abs(base["net_revenue"] - k["rev"]) < 0.5
    assert abs(base["contribution"] - k["con"]) < 0.5
    zero = S.compute(S.Drivers.from_totals({k_: 0 for k_ in tot}))
    assert zero["margin_pct"] is None and zero["contribution"] == 0


def test_scenario_directions_and_break_even():
    d = S.Drivers(units=1000, txns=400, list_price=1000, discount_rate=0.1, return_rate=0.05, cogs_per_unit=600,
                  shipping_per_txn=60, marketing_per_txn=40)
    b = S.compute(d)
    # hand calc: gross 1,000,000; disc 100,000; rev 900,000; ret 45,000; net 855,000; cogs 600,000; ship 24,000; mkt 16,000
    assert b["net_revenue"] == pytest.approx(855000) and b["contribution"] == pytest.approx(215000)
    assert S.compute(d, S.Adjustments(discount_pp=-5))["contribution"] == pytest.approx(215000 + 50000 * 0.95)
    assert S.compute(d, S.Adjustments(cogs_pct=10))["contribution"] == pytest.approx(155000)
    assert S.compute(d, S.Adjustments(volume_pct=10))["contribution"] == pytest.approx(236500)
    be = S.break_even(d)
    at_be = S.compute(S.Drivers(**{**d.__dict__, "discount_rate": be["max_discount_rate_pct"] / 100}))
    assert at_be["contribution"] == pytest.approx(0, abs=1e-6)
    at_be = S.compute(d, S.Adjustments(price_pct=be["price_change_to_break_even_pct"]))
    assert at_be["contribution"] == pytest.approx(0, abs=1e-6)
    w = S.compute(d, S.Adjustments(discount_pp=-50))["warnings"]
    assert w and "clipped" in w[0]


# ------------------------------------------------------------- filters
def test_build_where_uses_binds_only():
    from datetime import date
    f = A.Filters(start=date(2025, 1, 1), end=date(2025, 3, 31), category=["Audio", "x' OR 1=1 --"], region=["North"])
    where, params = A.build_where(f)
    assert "OR 1=1" not in where and where.count("?") == len(params) == 5
    assert "CATEGORY IN (?, ?)" in where and "REGION IN (?)" in where
    assert A.build_where(A.Filters()) == ("", [])


def test_aggregate_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        A.aggregate(None, None, ["CATEGORY; DROP TABLE X"])


# ------------------------------------------------------------- forecasting
def _monthly():
    df = V.read_csv(_sample_bytes())
    for c in V.NUMERIC_COLUMNS:
        df[c] = df[c].astype(float)
    df["SALE_MONTH"] = pd.to_datetime(df.transaction_date).dt.to_period("M").dt.to_timestamp()
    g = df.groupby("SALE_MONTH").agg(
        NET_REVENUE_INR=("net_revenue_inr", "sum"), GROSS_SALES_INR=("gross_revenue_inr", "sum"), UNITS=("quantity", "sum"),
        TXN_COUNT=("transaction_id", "nunique"), DISCOUNT_INR=("discount_amount_inr", "sum"), RETURNS_INR=("return_amount_inr", "sum"),
        COGS_INR=("cost_of_goods_inr", "sum"), SHIPPING_INR=("shipping_cost_inr", "sum"), MARKETING_INR=("marketing_cost_inr", "sum")).reset_index()
    g["CONTRIBUTION_INR"] = g.NET_REVENUE_INR - g.COGS_INR - g.SHIPPING_INR - g.MARKETING_INR
    return g


def test_forecast_input_validation():
    m = _monthly()
    assert F.validate_history(m) == []
    assert "at least" in F.validate_history(m.head(4))[0]
    assert "missing" in " ".join(F.validate_history(m.drop(index=5)))
    assert F.validate_history(m.iloc[0:0])
    with pytest.raises(ValueError):
        F.forecast(m.head(3), 3, "linear_trend")
    with pytest.raises(ValueError):
        F.forecast(m, 5, "linear_trend")


@pytest.mark.parametrize("h", F.HORIZONS)
def test_forecast_shapes_scenarios_and_order(h):
    res = F.forecast(_monthly(), h, "linear_trend")
    t = res["table"]
    assert len(t) == h and str(t.SALE_MONTH.iloc[0]) == "2026-01-01"
    assert (t.LOWER_95_INR <= t.FORECAST_NET_REVENUE_INR).all() and (t.FORECAST_NET_REVENUE_INR <= t.UPPER_95_INR).all()
    assert np.allclose(t.BASE_CASE_NET_REVENUE_INR, t.FORECAST_NET_REVENUE_INR)
    assert (t.CONSERVATIVE_CONTRIBUTION_INR < t.BASE_CASE_CONTRIBUTION_INR).all()
    assert (t.BASE_CASE_CONTRIBUTION_INR < t.OPTIMISTIC_CONTRIBUTION_INR).all()


def test_forecast_contribution_uses_drivers_not_flat_margin():
    m = _monthly()
    res = F.forecast(m, 3, "linear_trend")
    tail = m.tail(6)
    expected_margin = tail.CONTRIBUTION_INR.sum() / tail.NET_REVENUE_INR.sum()
    got = res["table"].BASE_CASE_CONTRIBUTION_INR / res["table"].BASE_CASE_NET_REVENUE_INR
    assert np.allclose(got, expected_margin, atol=0.002)


def test_backtest_without_ml():
    bt = F.evaluate(_monthly(), None, holdout=3, use_ml=False)
    assert set(bt.method) == {"recent_mean", "linear_trend"}
    assert (bt.MAE_INR > 0).all()
    assert F.metrics(np.array([100, 200]), np.array([110, 180]))["MAPE_PCT"] == pytest.approx(10.0)


# ------------------------------------------------------------- AI intent safety
def test_intent_validation_allow_lists():
    v = AI.validate_intent({"intent": "rank", "dimension": "product; drop", "metric": "x", "limit": 999})
    assert v["dimension"] == "product" and v["metric"] == "contribution" and v["limit"] == 20
    with pytest.raises(ValueError):
        AI.validate_intent({"intent": "sql", "query": "DROP TABLE"})
    w = AI.validate_intent({"intent": "what_if", "what_if": {"discount_pp": -5, "evil": "x", "price_pct": 1e9}})
    assert w["what_if"] == {"discount_pp": -5.0, "price_pct": 100.0}


@pytest.mark.parametrize("q,intent", [
    ("Which products contribute the most profit?", "rank"),
    ("Which categories have the highest return rate?", "rank"),
    ("What is reducing our contribution margin?", "drivers"),
    ("Which region should management investigate?", "investigate"),
    ("What would happen if discounts decreased by 5 percentage points?", "what_if"),
    ("What are the estimated contributions for the next six months?", "forecast"),
])
def test_keyword_fallback_examples(q, intent):
    v = AI.validate_intent(AI.keyword_intent(q))
    assert v["intent"] == intent
    if intent == "what_if":
        assert v["what_if"] == {"discount_pp": -5.0}
    if "categories" in q:
        assert v["dimension"] == "category" and v["metric"] == "return_rate"
    if "six" in q:
        assert v["horizon"] == 6


# ------------------------------------------------------------- reports
def _ctx():
    m = _monthly()
    k = {"NET_REVENUE_INR": m.NET_REVENUE_INR.sum(), "CONTRIBUTION_INR": m.CONTRIBUTION_INR.sum(), "MARGIN_PCT": 17.07,
         "TXN_COUNT": 1200, "AOV_INR": 8049.3, "RETURN_RATE_PCT": 5.67, "GROSS_SALES_INR": 1, "DISCOUNT_INR": 1, "RETURNS_INR": 1}
    fc = F.forecast(m, 6, "linear_trend")["table"]
    cat = pd.DataFrame({"CATEGORY": ["Audio", "Bags"], "NET_REVENUE_INR": [1e6, 5e5], "CONTRIBUTION_INR": [2e5, 1e5],
                        "MARGIN_PCT": [20, 20], "RETURN_RATE_PCT": [5, 6], "TXN_COUNT": [100, 50]})
    issues = pd.DataFrame([{"rank": 1, "issue": "High return rate", "dimension": "Product", "affected": "Webcam HD",
                            "evidence": "11.1% vs 5.7%", "explanation": "unverified", "action": "review", "impact_inr": 26975.0,
                            "impact_basis": "x", "effort": "Low", "uncertainty": "High", "kpi": "RR", "period": "3m", "score": 1, "evidence_n": 81}])
    d = S.Drivers(1000, 400, 1000, .1, .05, 600, 60, 40)
    return dict(title="Test report", report_type="Executive", filters={"Period": "2025-02-01 to 2025-11-30", "Region": "North"},
                coverage={"Transactions in scope": "1,200"}, kpis=k, monthly=m, category=cat, products=pd.DataFrame(),
                region=pd.DataFrame(), channel=pd.DataFrame(), issues=issues, forecast=fc, forecast_meta={"method": "Linear trend"},
                scenario=S.compare(d, S.SCENARIO_PRESETS["Optimistic"]), scenario_name="Optimistic",
                dq=pd.DataFrame([{"CHECK_NAME": "Duplicate transaction IDs", "SEVERITY": "ERROR", "STATUS": "PASS", "FAILED_ROWS": 0, "TOTAL_ROWS": 1200}]),
                assumptions=RG.DEFAULT_ASSUMPTIONS)


@pytest.mark.parametrize("rtype", RG.REPORT_TYPES)
def test_pdf_is_valid_and_contains_filters(rtype, tmp_path):
    pdf = RG.build_pdf({**_ctx(), "report_type": rtype})
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF") and len(pdf) > 3000
    out = tmp_path / f"r_{rtype}.pdf"
    out.write_bytes(pdf)
    from pypdf import PdfReader  # installed for tests
    text = "".join(p.extract_text() for p in PdfReader(str(out)).pages)
    assert "2025-02-01 to 2025-11-30" in text and "North" in text and rtype in text
    if rtype in ("Executive", "Forecast"):
        assert "ESTIMATE" in text


def test_inr_format():
    assert RG.inr(9659156.37) == "Rs 96,59,156"
    assert RG.inr(-1234.5, 2) == "-Rs 1,234.50"
    assert RG.inr(None) == "n/a" and RG.pct(float("nan")) == "n/a"


def test_csv_roundtrip():
    fc = F.forecast(_monthly(), 3, "linear_trend")["table"]
    back = pd.read_csv(io.BytesIO(fc.to_csv(index=False).encode()))
    assert len(back) == 3 and list(back.columns) == list(fc.columns)


# ------------------------------------------------------------- Snowflake integration (opt-in)
needs_sf = pytest.mark.skipif(not os.getenv("RUN_SNOWFLAKE_TESTS"), reason="set RUN_SNOWFLAKE_TESTS=1 and SNOWFLAKE_DEFAULT_CONNECTION_NAME")


@pytest.fixture(scope="module")
def sf():
    from src.snowflake_connection import get_session
    s = get_session()
    yield s
    s.close()


@needs_sf
def test_sf_kpis_reconcile_with_independent_sql(sf):
    k = A.kpis(sf, None)
    r = sf.sql("SELECT SUM(NET_REVENUE_INR), SUM(NET_REVENUE_INR-COST_OF_GOODS_INR-SHIPPING_COST_INR-MARKETING_COST_INR), "
               "COUNT(DISTINCT TRANSACTION_ID) FROM RETAIL_ANALYTICS.SALES.SALES_TRANSACTIONS").collect()[0]
    assert float(k["NET_REVENUE_INR"]) == pytest.approx(float(r[0]))
    assert float(k["CONTRIBUTION_INR"]) == pytest.approx(float(r[1]))
    assert int(k["TXN_COUNT"]) == r[2]


@needs_sf
def test_sf_filters_change_all_aggregates(sf):
    f = A.Filters(region=["North"])
    tot, m, c = A.kpis(sf, f), A.monthly(sf, f), A.by_dimension(sf, f, "CATEGORY")
    assert m.NET_REVENUE_INR.sum() == pytest.approx(tot["NET_REVENUE_INR"]) == pytest.approx(c.NET_REVENUE_INR.sum())
    assert tot["NET_REVENUE_INR"] < A.kpis(sf, None)["NET_REVENUE_INR"]
    tx = A.filtered_transactions(sf, f)
    assert set(tx.REGION) == {"North"} and len(tx) == tot["TXN_COUNT"]


@needs_sf
def test_sf_query_error_is_raised(sf):
    with pytest.raises(Exception):
        A.run(sf, "SELECT * FROM RETAIL_ANALYTICS.SALES.DOES_NOT_EXIST")
