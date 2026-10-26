"""Analytics queries. Every dashboard number comes from Snowflake through here.

Filters are turned into a WHERE clause with `?` bind parameters -- user values are
never concatenated into SQL. Column names come only from fixed allow-lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.snowflake_connection import fq

BASE_VIEW = fq("V_SALES_OVERVIEW")

# Allow-listed dimensions: UI label -> column
DIMENSIONS = {
    "Category": "CATEGORY",
    "Product": "PRODUCT_NAME",
    "Region": "REGION",
    "Sales channel": "SALES_CHANNEL",
    "Customer segment": "CUSTOMER_SEGMENT",
    "Payment method": "PAYMENT_METHOD",
    "Promotion": "PROMOTION_TYPE",
    "Month": "SALE_MONTH",
}

# The single definition of every KPI (same formulas as sql/04_analytics_views.sql)
MEASURES_SQL = """
  COUNT(DISTINCT TRANSACTION_ID)                                         AS TXN_COUNT,
  SUM(QUANTITY)                                                          AS UNITS,
  SUM(GROSS_SALES_INR)                                                   AS GROSS_SALES_INR,
  SUM(DISCOUNT_AMOUNT_INR)                                               AS DISCOUNT_INR,
  SUM(REVENUE_INR)                                                       AS REVENUE_INR,
  SUM(RETURN_AMOUNT_INR)                                                 AS RETURNS_INR,
  SUM(NET_REVENUE_INR)                                                   AS NET_REVENUE_INR,
  SUM(COST_OF_GOODS_INR)                                                 AS COGS_INR,
  SUM(SHIPPING_COST_INR)                                                 AS SHIPPING_INR,
  SUM(MARKETING_COST_INR)                                                AS MARKETING_INR,
  SUM(CONTRIBUTION_INR)                                                  AS CONTRIBUTION_INR,
  DIV0NULL(SUM(CONTRIBUTION_INR), SUM(NET_REVENUE_INR)) * 100            AS MARGIN_PCT,
  DIV0NULL(SUM(NET_REVENUE_INR), COUNT(DISTINCT TRANSACTION_ID))         AS AOV_INR,
  DIV0NULL(SUM(REVENUE_INR), SUM(QUANTITY))                              AS AVG_SELLING_PRICE_INR,
  DIV0NULL(SUM(DISCOUNT_AMOUNT_INR), SUM(GROSS_SALES_INR)) * 100         AS EFFECTIVE_DISCOUNT_PCT,
  COUNT(DISTINCT IFF(RETURNED, TRANSACTION_ID, NULL))                    AS RETURNED_TXNS,
  DIV0NULL(COUNT(DISTINCT IFF(RETURNED, TRANSACTION_ID, NULL)), COUNT(DISTINCT TRANSACTION_ID)) * 100 AS RETURN_RATE_PCT,
  COUNT_IF(CONTRIBUTION_INR < 0)                                         AS NEG_CONTRIBUTION_TXNS,
  COUNT_IF(CONTRIBUTION_INR IS NULL)                                     AS MISSING_COST_ROWS
"""


@dataclass
class Filters:
    start: date | None = None
    end: date | None = None
    category: list[str] = field(default_factory=list)
    region: list[str] = field(default_factory=list)
    sales_channel: list[str] = field(default_factory=list)
    customer_segment: list[str] = field(default_factory=list)
    promotion_type: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, str]:
        f = lambda v: ", ".join(v) if v else "All"
        return {
            "Period": f"{self.start or 'start'} to {self.end or 'end'}",
            "Category": f(self.category),
            "Region": f(self.region),
            "Sales channel": f(self.sales_channel),
            "Customer segment": f(self.customer_segment),
            "Promotion": f(self.promotion_type),
        }


_LIST_FILTER_COLS = {
    "category": "CATEGORY",
    "region": "REGION",
    "sales_channel": "SALES_CHANNEL",
    "customer_segment": "CUSTOMER_SEGMENT",
    "promotion_type": "PROMOTION_TYPE",
}


def build_where(filters: Filters | None) -> tuple[str, list]:
    """Return (' WHERE ...', params) using qmark binds. Pure function (unit tested)."""
    if filters is None:
        return "", []
    clauses, params = [], []
    if filters.start:
        clauses.append("TRANSACTION_DATE >= ?")
        params.append(str(filters.start))
    if filters.end:
        clauses.append("TRANSACTION_DATE <= ?")
        params.append(str(filters.end))
    for attr, col in _LIST_FILTER_COLS.items():
        vals = getattr(filters, attr)
        if vals:
            clauses.append(f"{col} IN ({', '.join('?' for _ in vals)})")
            params.extend(vals)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def run(session, sql: str, params: list | None = None) -> pd.DataFrame:
    df = session.sql(sql, params=params or None).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    # Snowflake NUMBER -> Decimal/object; make numeric columns float for charts
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum() and df[c].notna().any():
                df[c] = conv
    return df


def aggregate(session, filters: Filters | None, group_by: list[str] | None = None) -> pd.DataFrame:
    group_by = group_by or []
    allowed = set(DIMENSIONS.values())
    bad = [g for g in group_by if g not in allowed]
    if bad:
        raise ValueError(f"Dimension not allowed: {bad}")
    where, params = build_where(filters)
    sel = (", ".join(group_by) + ",") if group_by else ""
    grp = (" GROUP BY " + ", ".join(group_by) + " ORDER BY " + ", ".join(group_by)) if group_by else ""
    return run(session, f"SELECT {sel} {MEASURES_SQL} FROM {BASE_VIEW}{where}{grp}", params)


def kpis(session, filters: Filters | None) -> dict:
    df = aggregate(session, filters)
    return df.iloc[0].to_dict() if len(df) else {}


def monthly(session, filters: Filters | None) -> pd.DataFrame:
    return aggregate(session, filters, ["SALE_MONTH"])


def by_dimension(session, filters: Filters | None, *dims: str) -> pd.DataFrame:
    return aggregate(session, filters, list(dims))


def filtered_transactions(session, filters: Filters | None, limit: int | None = None) -> pd.DataFrame:
    where, params = build_where(filters)
    lim = f" LIMIT {int(limit)}" if limit else ""
    return run(session, f"SELECT * EXCLUDE (UPLOAD_ID) FROM {BASE_VIEW}{where} ORDER BY TRANSACTION_DATE, TRANSACTION_ID{lim}", params)


def product_return_trend(session, filters: Filters | None) -> pd.DataFrame:
    """Return-rate change, second half vs first half of the filtered period, per product."""
    where, params = build_where(filters)
    sql = f"""
    WITH b AS (SELECT * FROM {BASE_VIEW}{where}),
    mid AS (SELECT DATEADD(DAY, DATEDIFF(DAY, MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE)) / 2, MIN(TRANSACTION_DATE)) m FROM b)
    SELECT PRODUCT_NAME,
      COUNT_IF(TRANSACTION_DATE <= m) AS TXNS_FIRST_HALF,
      COUNT_IF(TRANSACTION_DATE > m)  AS TXNS_SECOND_HALF,
      DIV0NULL(COUNT_IF(RETURNED AND TRANSACTION_DATE <= m), COUNT_IF(TRANSACTION_DATE <= m)) * 100 AS RR_FIRST_HALF_PCT,
      DIV0NULL(COUNT_IF(RETURNED AND TRANSACTION_DATE > m),  COUNT_IF(TRANSACTION_DATE > m))  * 100 AS RR_SECOND_HALF_PCT
    FROM b, mid GROUP BY PRODUCT_NAME
    """
    df = run(session, sql, params)
    if len(df):
        df["RR_CHANGE_PP"] = df["RR_SECOND_HALF_PCT"] - df["RR_FIRST_HALF_PCT"]
    return df.sort_values("RR_CHANGE_PP", ascending=False) if len(df) else df


def discount_bands(session, filters: Filters | None) -> pd.DataFrame:
    where, params = build_where(filters)
    sql = f"""SELECT DISCOUNT_PCT, {MEASURES_SQL} FROM {BASE_VIEW}{where} GROUP BY DISCOUNT_PCT ORDER BY DISCOUNT_PCT"""
    return run(session, sql, params)


def filter_options(session) -> dict[str, list]:
    df = run(session, f"""SELECT
        ARRAY_AGG(DISTINCT CATEGORY) c, ARRAY_AGG(DISTINCT REGION) r, ARRAY_AGG(DISTINCT SALES_CHANNEL) ch,
        ARRAY_AGG(DISTINCT CUSTOMER_SEGMENT) s, ARRAY_AGG(DISTINCT PROMOTION_TYPE) p,
        MIN(TRANSACTION_DATE) mn, MAX(TRANSACTION_DATE) mx FROM {BASE_VIEW}""")
    import json

    row = df.iloc[0]
    lst = lambda v: sorted(json.loads(v)) if isinstance(v, str) else sorted(v or [])
    return {
        "category": lst(row["C"]), "region": lst(row["R"]), "sales_channel": lst(row["CH"]),
        "customer_segment": lst(row["S"]), "promotion_type": lst(row["P"]),
        "min_date": pd.to_datetime(row["MN"]).date() if row["MN"] is not None else None,
        "max_date": pd.to_datetime(row["MX"]).date() if row["MX"] is not None else None,
    }


def data_quality_latest(session) -> pd.DataFrame:
    return run(session, f"""SELECT CHECKED_AT, CHECK_NAME, SEVERITY, STATUS, FAILED_ROWS, TOTAL_ROWS,
        ARRAY_TO_STRING(SAMPLE_IDS, ', ') AS SAMPLE_IDS, DETAILS
      FROM {fq('DATA_QUALITY_RESULTS')}
      WHERE CHECK_RUN_ID = (SELECT CHECK_RUN_ID FROM {fq('DATA_QUALITY_RESULTS')} ORDER BY CHECKED_AT DESC LIMIT 1)
      ORDER BY CASE STATUS WHEN 'FAIL' THEN 0 WHEN 'WARN' THEN 1 WHEN 'INFO' THEN 2 ELSE 3 END, CHECK_NAME""")


def run_data_quality(session) -> pd.DataFrame:
    session.sql(f"CALL {fq('RUN_DATA_QUALITY_CHECKS')}()").collect()
    return data_quality_latest(session)


def upload_history(session) -> pd.DataFrame:
    return run(session, f"SELECT * FROM {fq('UPLOAD_HISTORY')} ORDER BY UPLOADED_AT DESC LIMIT 100")


def reconcile(session) -> pd.DataFrame:
    """Independent check: base table vs analytic view totals."""
    return run(session, f"""
      SELECT 'Base table' AS SOURCE, COUNT(DISTINCT TRANSACTION_ID) TXN_COUNT, SUM(NET_REVENUE_INR) NET_REVENUE_INR,
             SUM(NET_REVENUE_INR - COST_OF_GOODS_INR - SHIPPING_COST_INR - MARKETING_COST_INR) CONTRIBUTION_INR
        FROM {fq('SALES_TRANSACTIONS')}
      UNION ALL SELECT 'V_MONTHLY_SALES', SUM(TXN_COUNT), SUM(NET_REVENUE_INR), SUM(CONTRIBUTION_INR) FROM {fq('V_MONTHLY_SALES')}
      UNION ALL SELECT 'V_CATEGORY_PROFITABILITY', SUM(TXN_COUNT), SUM(NET_REVENUE_INR), SUM(CONTRIBUTION_INR) FROM {fq('V_CATEGORY_PROFITABILITY')}
      UNION ALL SELECT 'V_REGIONAL_PERFORMANCE', SUM(TXN_COUNT), SUM(NET_REVENUE_INR), SUM(CONTRIBUTION_INR) FROM {fq('V_REGIONAL_PERFORMANCE')}
    """)
