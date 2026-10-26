"""CSV validation and column mapping (pure pandas, no Snowflake dependency).

Nothing is ever silently dropped: every row gets a list of issues, and the
caller decides (with the user) whether to load valid rows only, or reject the file.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id", "transaction_date", "product_name", "category", "region",
    "sales_channel", "customer_segment", "quantity", "unit_price_inr", "discount_pct",
    "gross_revenue_inr", "discount_amount_inr", "revenue_inr", "cost_of_goods_inr",
    "shipping_cost_inr", "marketing_cost_inr", "returned", "return_amount_inr",
    "net_revenue_inr", "net_profit_inr", "payment_method", "promotion_type",
]
NUMERIC_COLUMNS = [
    "quantity", "unit_price_inr", "discount_pct", "gross_revenue_inr", "discount_amount_inr",
    "revenue_inr", "cost_of_goods_inr", "shipping_cost_inr", "marketing_cost_inr",
    "return_amount_inr", "net_revenue_inr", "net_profit_inr",
]
# Columns whose absence makes the row unusable
CRITICAL_COLUMNS = ["transaction_id", "transaction_date", "product_name", "category", "region",
                    "sales_channel", "quantity", "unit_price_inr", "net_revenue_inr"]
# Cost columns: if missing, contribution becomes "unavailable" (never invented)
COST_COLUMNS = ["cost_of_goods_inr", "shipping_cost_inr", "marketing_cost_inr"]
# Derivable columns: can be computed from others if the file omits them
DERIVABLE = {"gross_revenue_inr", "discount_amount_inr", "revenue_inr", "return_amount_inr",
             "net_profit_inr", "returned", "discount_pct", "promotion_type", "payment_method",
             "customer_segment"}

TRUE_VALUES = {"true", "t", "yes", "y", "1"}
FALSE_VALUES = {"false", "f", "no", "n", "0", ""}
TOL = 0.05  # INR tolerance for arithmetic consistency checks


def read_csv(data: bytes | str) -> pd.DataFrame:
    """Read everything as text so we can validate types ourselves.
    keep_default_na=False preserves literal 'None' (= no promotion) etc."""
    buf = io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data)
    return pd.read_csv(buf, dtype=str, keep_default_na=False, na_values=[""])


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("inr", ""))


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Map required column -> source column using normalised-name matching."""
    aliases = {
        "transaction_id": ["txnid", "orderid", "id", "invoiceno", "transactionid"],
        "transaction_date": ["date", "orderdate", "txndate", "transactiondate"],
        "product_name": ["product", "item", "productname", "sku"],
        "unit_price_inr": ["price", "unitprice"],
        "quantity": ["qty", "units", "quantity"],
        "discount_pct": ["discount", "discountpercent", "discountpct"],
        "cost_of_goods_inr": ["cogs", "cost", "costofgoods", "costofgoodssold"],
        "returned": ["isreturned", "return", "returned"],
        "sales_channel": ["channel", "saleschannel"],
        "customer_segment": ["segment", "customersegment"],
    }
    src = {_norm(c): c for c in columns}
    out = {}
    for req in REQUIRED_COLUMNS:
        cands = [_norm(req)] + aliases.get(req, [])
        out[req] = next((src[c] for c in cands if c in src), None)
    return out


def apply_mapping(raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    for req in REQUIRED_COLUMNS:
        srccol = mapping.get(req)
        df[req] = raw[srccol] if srccol and srccol in raw.columns else np.nan
    df.insert(0, "_source_row", raw.index + 2)  # +2 = header + 1-based line number
    return df


def _to_bool(s: pd.Series) -> pd.Series:
    low = s.fillna("").astype(str).str.strip().str.lower()
    return low.map(lambda v: True if v in TRUE_VALUES else (False if v in FALSE_VALUES else None))


def coerce_and_derive(df: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, list[str]]:
    """Convert types and fill derivable columns. Returns (typed_df, notes)."""
    notes = []
    out = df.copy()
    for c in NUMERIC_COLUMNS:
        out[c + "__raw"] = out[c]
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["transaction_date__raw"] = out["transaction_date"]
    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce", format="mixed").dt.date
    out["returned__raw"] = out["returned"]
    out["returned"] = _to_bool(out["returned"])

    m = lambda c: mapping.get(c) is None
    if m("gross_revenue_inr"):
        out["gross_revenue_inr"] = (out["unit_price_inr"] * out["quantity"]).round(2)
        notes.append("gross_revenue_inr derived as unit_price_inr x quantity")
    if m("discount_pct"):
        out["discount_pct"] = 0.0
        notes.append("discount_pct missing - assumed 0")
    if m("discount_amount_inr"):
        out["discount_amount_inr"] = (out["gross_revenue_inr"] * out["discount_pct"] / 100).round(2)
        notes.append("discount_amount_inr derived from gross x discount_pct")
    if m("revenue_inr"):
        out["revenue_inr"] = (out["gross_revenue_inr"] - out["discount_amount_inr"]).round(2)
        notes.append("revenue_inr derived as gross - discount")
    if m("return_amount_inr"):
        out["return_amount_inr"] = (out["revenue_inr"] - out["net_revenue_inr"]).clip(lower=0).round(2)
        notes.append("return_amount_inr derived as revenue - net_revenue")
    if m("returned"):
        out["returned"] = out["return_amount_inr"] > 0
        notes.append("returned derived from return_amount_inr > 0")
    for c in ("promotion_type", "payment_method", "customer_segment"):
        if m(c):
            out[c] = "Unknown" if c != "promotion_type" else "None"
            notes.append(f"{c} missing - set to '{out[c].iloc[0] if len(out) else ''}'")
    missing_costs = [c for c in COST_COLUMNS if m(c)]
    if missing_costs:
        notes.append(f"Cost columns missing ({', '.join(missing_costs)}): contribution will be UNAVAILABLE (left NULL), not estimated")
    if m("net_profit_inr") and not missing_costs:
        out["net_profit_inr"] = (out["net_revenue_inr"] - out["cost_of_goods_inr"] - out["shipping_cost_inr"] - out["marketing_cost_inr"]).round(2)
        notes.append("net_profit_inr derived as net revenue - COGS - shipping - marketing")
    return out, notes


def validate(df: pd.DataFrame, mapping: dict[str, str | None], existing_ids: set[str] | None = None) -> dict:
    """Validate a mapped+typed frame. Returns dict with per-row issues and a summary."""
    issues: dict[int, list[str]] = {i: [] for i in df.index}
    warn: dict[int, list[str]] = {i: [] for i in df.index}

    def flag(mask, msg, bucket=issues):
        for i in df.index[mask.fillna(False).astype(bool)]:
            bucket[i].append(msg)

    unmapped_critical = [c for c in CRITICAL_COLUMNS if mapping.get(c) is None]
    for c in CRITICAL_COLUMNS:
        if c in unmapped_critical:
            continue
        raw = df[c + "__raw"] if c + "__raw" in df else df[c]
        flag(raw.isna() | (raw.astype(str).str.strip() == ""), f"missing {c}")

    for c in NUMERIC_COLUMNS:
        if mapping.get(c) is None:
            continue
        raw = df[c + "__raw"]
        flag(raw.notna() & (raw.astype(str).str.strip() != "") & df[c].isna(), f"{c} not numeric")
    if mapping.get("transaction_date"):
        flag(df["transaction_date__raw"].notna() & df["transaction_date"].isna(), "invalid transaction_date")
    if mapping.get("returned"):
        flag(df["returned"].isna(), "returned not a boolean")

    flag(df["quantity"] <= 0, "quantity must be > 0")
    flag((df["discount_pct"] < 0) | (df["discount_pct"] > 100), "discount_pct outside 0-100")
    for c in ["unit_price_inr", "cost_of_goods_inr", "shipping_cost_inr", "marketing_cost_inr", "return_amount_inr"]:
        flag(df[c] < 0, f"negative {c}")
    flag(df["return_amount_inr"] > df["revenue_inr"] + TOL, "return_amount exceeds revenue")
    today = pd.Timestamp.today().date()
    flag(df["transaction_date"].map(lambda d: d is not None and not pd.isna(d) and d > today), "future transaction_date", warn)

    # Arithmetic consistency -> warnings (data kept as-is)
    flag((df["unit_price_inr"] * df["quantity"] - df["gross_revenue_inr"]).abs() > TOL, "gross <> price x qty", warn)
    flag((df["gross_revenue_inr"] - df["discount_amount_inr"] - df["revenue_inr"]).abs() > TOL, "revenue <> gross - discount", warn)
    flag((df["revenue_inr"] - df["return_amount_inr"] - df["net_revenue_inr"]).abs() > TOL, "net revenue <> revenue - returns", warn)
    flag(df["returned"].eq(True) & (df["return_amount_inr"] == 0), "returned=TRUE but return_amount=0", warn)

    ids = df["transaction_id"].astype(str).str.strip()
    dup_in_file = ids.duplicated(keep=False) & df["transaction_id"].notna()
    flag(dup_in_file, "duplicate transaction_id in file")
    if existing_ids:
        flag(ids.isin(existing_ids), "transaction_id already exists in target table", warn)

    df = df.copy()
    df["_errors"] = [", ".join(issues[i]) for i in df.index]
    df["_warnings"] = [", ".join(warn[i]) for i in df.index]
    df["_valid"] = df["_errors"] == ""
    counts: dict[str, int] = {}
    for lst in list(issues.values()) + list(warn.values()):
        for m in lst:
            counts[m] = counts.get(m, 0) + 1
    summary = {
        "rows_in_file": int(len(df)),
        "valid_rows": int(df["_valid"].sum()),
        "invalid_rows": int((~df["_valid"]).sum()),
        "rows_with_warnings": int((df["_warnings"] != "").sum()),
        "duplicate_ids_in_file": int(dup_in_file.sum()),
        "unmapped_critical_columns": unmapped_critical,
        "issue_counts": counts,
    }
    return {"frame": df, "summary": summary, "blocking": bool(unmapped_critical)}


def clean_for_load(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 22 business columns, typed for Snowflake (uppercase names)."""
    out = df[REQUIRED_COLUMNS].copy()
    out["transaction_date"] = pd.to_datetime(out["transaction_date"])
    out["returned"] = out["returned"].astype("boolean")
    out["quantity"] = out["quantity"].astype("Int64")
    out.columns = [c.upper() for c in out.columns]
    return out
