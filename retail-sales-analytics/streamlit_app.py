"""Retail Sales Analytics & AI Insights - Streamlit app (local or Streamlit in Snowflake)."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ai_assistant as AI
from src import analytics as A
from src import data_loader as L
from src import forecasting as F
from src import recommendations as R
from src import report_generator as RG
from src import scenarios as S
from src import validation as V
from src.snowflake_connection import get_session

st.set_page_config(page_title="Retail Sales Analytics", page_icon=":bar_chart:", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_resource
def session():
    return get_session()


def inr(v, dec=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return RG.inr(v, dec).replace("Rs ", "₹")


def pct(v, dec=1):
    return RG.pct(v, dec)


@st.cache_data(ttl=600, show_spinner="Querying Snowflake…")
def agg(f: A.Filters, dims: tuple = ()) -> pd.DataFrame:
    return A.aggregate(session(), f, list(dims))


@st.cache_data(ttl=600, show_spinner=False)
def options():
    return A.filter_options(session())


@st.cache_data(ttl=600, show_spinner="Running rules over Snowflake results…")
def issues_df(f: A.Filters) -> pd.DataFrame:
    return R.to_frame(R.detect_issues(session(), f))


@st.cache_data(ttl=600, show_spinner=False)
def dq_latest() -> pd.DataFrame:
    return A.data_quality_latest(session())


@st.cache_data(ttl=600, show_spinner=False)
def cached(name: str, f: A.Filters):
    return getattr(A, name)(session(), f)


@st.cache_data(ttl=3600, show_spinner="Back-testing forecast methods…")
def backtest(f: A.Filters, use_ml: bool):
    return F.evaluate(agg(f, ("SALE_MONTH",)), session(), holdout=3, use_ml=use_ml)


@st.cache_data(ttl=3600, show_spinner="Forecasting…")
def run_forecast(f: A.Filters, horizon: int, method: str, scen_json: str):
    scen = {k: S.Adjustments(**v) for k, v in json.loads(scen_json).items()}
    return F.forecast(agg(f, ("SALE_MONTH",)), horizon, method, session(), scen)


MONEY_COLS = ["NET_REVENUE_INR", "CONTRIBUTION_INR", "GROSS_SALES_INR", "DISCOUNT_INR", "RETURNS_INR", "COGS_INR",
              "SHIPPING_INR", "MARKETING_INR", "AOV_INR", "AVG_SELLING_PRICE_INR", "REVENUE_INR"]
PCT_COLS = ["MARGIN_PCT", "RETURN_RATE_PCT", "EFFECTIVE_DISCOUNT_PCT"]


def col_config(df):
    cfg = {}
    for c in df.columns:
        if c in MONEY_COLS or c.endswith("_INR"):
            cfg[c] = st.column_config.NumberColumn(c.replace("_INR", " (₹)").replace("_", " ").title(), format="₹ %.0f")
        elif c in PCT_COLS or c.endswith("_PCT") or c.endswith("_PP"):
            cfg[c] = st.column_config.NumberColumn(c.replace("_", " ").title(), format="%.1f")
    return cfg


def table(df, **kw):
    st.dataframe(df, column_config=col_config(df), hide_index=True, use_container_width=True, **kw)


def empty_guard(df, msg="No data for the selected filters. Widen the filters in the sidebar."):
    if df is None or len(df) == 0:
        st.info(msg)
        return True
    return False


# ---------------------------------------------------------------- sidebar filters
def sidebar_filters() -> A.Filters:
    try:
        o = options()
    except Exception as e:
        st.sidebar.error(f"Cannot reach Snowflake: {e}")
        st.stop()
    st.sidebar.header("Filters")
    if o["min_date"] is None:
        st.sidebar.info("No data loaded yet - use **Upload Data**.")
        return A.Filters()
    dr = st.sidebar.date_input("Date range", (o["min_date"], o["max_date"]), min_value=o["min_date"], max_value=o["max_date"])
    start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (o["min_date"], o["max_date"]))
    f = A.Filters(
        start=start, end=end,
        category=st.sidebar.multiselect("Category", o["category"]),
        region=st.sidebar.multiselect("Region", o["region"]),
        sales_channel=st.sidebar.multiselect("Sales channel", o["sales_channel"]),
        customer_segment=st.sidebar.multiselect("Customer segment", o["customer_segment"]),
        promotion_type=st.sidebar.multiselect("Promotion", o["promotion_type"]),
    )
    if st.sidebar.button("Refresh data", help="Clear cached query results"):
        st.cache_data.clear()
        st.rerun()
    return f


# ---------------------------------------------------------------- pages
def kpi_row(k):
    c = st.columns(6)
    c[0].metric("Net revenue (actual)", inr(k.get("NET_REVENUE_INR")))
    c[1].metric("Est. contribution", inr(k.get("CONTRIBUTION_INR")), help="Net revenue − COGS − shipping − marketing (not full accounting profit)")
    c[2].metric("Contribution margin", pct(k.get("MARGIN_PCT")))
    c[3].metric("Transactions", f"{int(k.get('TXN_COUNT') or 0):,}")
    c[4].metric("Avg order value", inr(k.get("AOV_INR")))
    c[5].metric("Return rate", pct(k.get("RETURN_RATE_PCT")))


def page_overview(f):
    st.title("Executive Overview")
    k = agg(f).iloc[0].to_dict()
    if not k.get("TXN_COUNT"):
        st.info("No transactions match the filters.")
        return
    kpi_row(k)
    dq = dq_latest()
    bad = dq[dq.STATUS.isin(["FAIL", "WARN"])] if len(dq) else dq
    if len(dq) == 0:
        st.warning("Data-quality checks have not been run yet (see **Data Quality**).")
    elif len(bad):
        st.error(f"Data quality: {len(bad)} check(s) failing or warning - see **Data Quality**.")
    else:
        st.success(f"Data quality: all {len(dq[dq.STATUS == 'PASS'])} checks passing (last run {pd.to_datetime(dq.CHECKED_AT.iloc[0]):%Y-%m-%d %H:%M}).")
    if k.get("NEG_CONTRIBUTION_TXNS"):
        st.caption(f"⚠ {int(k['NEG_CONTRIBUTION_TXNS'])} transactions have negative contribution.")

    m = agg(f, ("SALE_MONTH",))
    fig = go.Figure()
    fig.add_bar(x=m.SALE_MONTH, y=m.NET_REVENUE_INR, name="Net revenue")
    fig.add_scatter(x=m.SALE_MONTH, y=m.CONTRIBUTION_INR, name="Est. contribution", mode="lines+markers")
    fig.update_layout(title="Monthly net revenue and estimated contribution (₹)", height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    cat, reg = agg(f, ("CATEGORY",)), agg(f, ("REGION",))
    c1, c2 = st.columns(2)
    tc = cat.sort_values("CONTRIBUTION_INR", ascending=False).iloc[0]
    tr = reg.sort_values("CONTRIBUTION_INR", ascending=False).iloc[0]
    c1.metric("Top category (by contribution)", tc.CATEGORY, f"{inr(tc.CONTRIBUTION_INR)} · {pct(tc.MARGIN_PCT)} margin", delta_color="off")
    c2.metric("Highest-contribution region", tr.REGION, f"{inr(tr.CONTRIBUTION_INR)} · {pct(tr.MARGIN_PCT)} margin", delta_color="off")
    iss = issues_df(f)
    if len(iss):
        st.subheader("Top issues to review")
        table(iss[["rank", "issue", "affected", "evidence", "impact_inr", "uncertainty"]].head(5))


def page_products(f):
    st.title("Product & Category Analysis")
    p, c = agg(f, ("PRODUCT_NAME", "CATEGORY")), agg(f, ("CATEGORY",))
    if empty_guard(p):
        return
    fig = px.bar(c.sort_values("NET_REVENUE_INR"), x=["NET_REVENUE_INR", "CONTRIBUTION_INR"], y="CATEGORY", barmode="group",
                 orientation="h", title="Net revenue vs estimated contribution by category (₹)", labels={"value": "₹", "variable": ""})
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Best-selling (units)")
        table(p.nlargest(5, "UNITS")[["PRODUCT_NAME", "UNITS", "NET_REVENUE_INR"]])
    with c2:
        st.subheader("Highest contribution")
        table(p.nlargest(5, "CONTRIBUTION_INR")[["PRODUCT_NAME", "CONTRIBUTION_INR", "MARGIN_PCT"]])
    with c3:
        st.subheader("Lowest contribution")
        table(p.nsmallest(5, "CONTRIBUTION_INR")[["PRODUCT_NAME", "CONTRIBUTION_INR", "MARGIN_PCT"]])

    st.subheader("High sales but weak contribution")
    k = agg(f).iloc[0]
    weak = p[(p.NET_REVENUE_INR >= p.NET_REVENUE_INR.median()) & (p.MARGIN_PCT < k.MARGIN_PCT)]
    fig = px.scatter(p, x="NET_REVENUE_INR", y="MARGIN_PCT", size="UNITS", color="CATEGORY", hover_name="PRODUCT_NAME",
                     title="Revenue vs margin (bubble = units)", labels={"NET_REVENUE_INR": "Net revenue (₹)", "MARGIN_PCT": "Margin %"})
    fig.add_hline(y=k.MARGIN_PCT, line_dash="dot", annotation_text=f"Overall margin {k.MARGIN_PCT:.1f}%")
    fig.add_vline(x=p.NET_REVENUE_INR.median(), line_dash="dot", annotation_text="Median revenue")
    st.plotly_chart(fig, use_container_width=True)
    if len(weak):
        table(weak[["PRODUCT_NAME", "CATEGORY", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "EFFECTIVE_DISCOUNT_PCT", "RETURN_RATE_PCT"]])
    else:
        st.caption("No product is both above-median revenue and below the overall margin.")

    st.subheader("Return-rate trend by product (2nd half vs 1st half of selected period)")
    tr = cached("product_return_trend", f)
    table(tr)
    st.caption("Changes on small samples (<30 txns per half) are noisy - treat as signals, not conclusions.")
    st.subheader("All products")
    table(p[["PRODUCT_NAME", "CATEGORY", "UNITS", "AVG_SELLING_PRICE_INR", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT",
             "RETURN_RATE_PCT", "TXN_COUNT"]].sort_values("CONTRIBUTION_INR", ascending=False))


def page_regions(f):
    st.title("Regional & Channel Analysis")
    rc = agg(f, ("REGION", "SALES_CHANNEL"))
    if empty_guard(rc):
        return
    reg, ch = agg(f, ("REGION",)), agg(f, ("SALES_CHANNEL",))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(rc, x="REGION", y="NET_REVENUE_INR", color="SALES_CHANNEL", title="Net revenue by region & channel (₹)"), use_container_width=True)
    c2.plotly_chart(px.bar(rc, x="REGION", y="MARGIN_PCT", color="SALES_CHANNEL", barmode="group", title="Contribution margin % by region & channel"), use_container_width=True)
    k = agg(f).iloc[0]
    st.subheader("Regions (volume-adjusted)")
    reg["MARGIN_GAP_PP"] = reg.MARGIN_PCT - k.MARGIN_PCT
    reg["REVENUE_SHARE_PCT"] = reg.NET_REVENUE_INR / reg.NET_REVENUE_INR.sum() * 100
    table(reg[["REGION", "TXN_COUNT", "REVENUE_SHARE_PCT", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "MARGIN_GAP_PP", "AOV_INR",
               "RETURN_RATE_PCT", "EFFECTIVE_DISCOUNT_PCT"]].sort_values("MARGIN_GAP_PP"))
    under = reg[(reg.MARGIN_GAP_PP < -1.5) & (reg.TXN_COUNT >= 30)]
    if len(under):
        st.warning("Underperforming (margin > 1.5 pp below overall, ≥30 txns): " + ", ".join(under.REGION) +
                   ". Comparisons use margin % and AOV so regions of different size are comparable; check product mix before concluding.")
    st.subheader("Channels")
    table(ch[["SALES_CHANNEL", "TXN_COUNT", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "AOV_INR", "RETURN_RATE_PCT", "SHIPPING_INR"]])
    c1, c2, c3 = st.columns(3)
    for col, dim, title in [(c1, "CUSTOMER_SEGMENT", "Customer segments"), (c2, "PAYMENT_METHOD", "Payment methods"), (c3, "PROMOTION_TYPE", "Promotions")]:
        d = agg(f, (dim,))
        col.plotly_chart(px.bar(d, x=dim, y="CONTRIBUTION_INR", color="MARGIN_PCT", title=f"{title}: contribution (colour = margin %)",
                                color_continuous_scale="RdYlGn"), use_container_width=True)
    st.subheader("Promotion effectiveness")
    pr = agg(f, ("PROMOTION_TYPE",))
    pr["UNITS_PER_TXN"] = pr.UNITS / pr.TXN_COUNT
    table(pr[["PROMOTION_TYPE", "TXN_COUNT", "UNITS_PER_TXN", "AOV_INR", "EFFECTIVE_DISCOUNT_PCT", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "RETURN_RATE_PCT"]])
    st.caption("Promotion comparisons are observational: product mix differs between groups and there is no control group, so differences are correlations.")


def page_costs(f):
    st.title("Discounts, Returns & Costs")
    b = cached("discount_bands", f)
    if empty_guard(b):
        return
    c1, c2 = st.columns(2)
    fig = px.bar(b, x="DISCOUNT_PCT", y="MARGIN_PCT", text="TXN_COUNT", title="Contribution margin % by discount level (label = txns)",
                 labels={"DISCOUNT_PCT": "Discount %", "MARGIN_PCT": "Margin %"})
    c1.plotly_chart(fig, use_container_width=True)
    if len(b) >= 3:
        r = b[["DISCOUNT_PCT", "MARGIN_PCT"]].corr().iloc[0, 1]
        c1.caption(f"Correlation (discount %, margin %) across bands: {r:.2f}. Lower margin at higher discount is partly mechanical "
                   "(lower price, same cost); this data cannot show whether discounts generated extra volume.")
    k = agg(f).iloc[0]
    burden = pd.DataFrame({"Cost": ["COGS", "Shipping", "Marketing", "Discounts", "Returns"],
                           "INR": [k.COGS_INR, k.SHIPPING_INR, k.MARKETING_INR, k.DISCOUNT_INR, k.RETURNS_INR]})
    burden["% of gross sales"] = burden.INR / k.GROSS_SALES_INR * 100
    c2.plotly_chart(px.bar(burden, x="Cost", y="% of gross sales", title="Cost & leakage burden (% of gross sales)"), use_container_width=True)

    st.subheader("Return rates")
    tabs = st.tabs(["Product", "Category", "Region", "Channel"])
    for t, dim in zip(tabs, ["PRODUCT_NAME", "CATEGORY", "REGION", "SALES_CHANNEL"]):
        with t:
            d = agg(f, (dim,)).sort_values("RETURN_RATE_PCT", ascending=False)
            fig = px.bar(d, x=dim, y="RETURN_RATE_PCT", text="RETURNED_TXNS", title="Return rate % (label = returned txns)")
            fig.add_hline(y=k.RETURN_RATE_PCT, line_dash="dot", annotation_text="overall")
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Shipping & marketing per transaction over time")
    m = agg(f, ("SALE_MONTH",))
    m["SHIPPING_PER_TXN"] = m.SHIPPING_INR / m.TXN_COUNT
    m["MARKETING_PER_TXN"] = m.MARKETING_INR / m.TXN_COUNT
    st.plotly_chart(px.line(m, x="SALE_MONTH", y=["SHIPPING_PER_TXN", "MARKETING_PER_TXN"], markers=True, labels={"value": "₹ per txn"}),
                    use_container_width=True)
    st.subheader("Promotions that may be reducing contribution")
    pr = agg(f, ("PROMOTION_TYPE",))
    base = pr[pr.PROMOTION_TYPE == "No Promotion"]
    if len(base):
        pr["MARGIN_VS_NO_PROMO_PP"] = pr.MARGIN_PCT - base.MARGIN_PCT.iloc[0]
    table(pr[[c for c in ["PROMOTION_TYPE", "TXN_COUNT", "EFFECTIVE_DISCOUNT_PCT", "MARGIN_PCT", "MARGIN_VS_NO_PROMO_PP", "CONTRIBUTION_INR"] if c in pr]])
    st.subheader("Anomalies: negative-contribution transactions")
    tx = cached("filtered_transactions", f)
    neg = tx[tx.CONTRIBUTION_INR < 0]
    st.write(f"{len(neg)} transactions ({len(neg[neg.RETURNED])} returned, {len(neg[~neg.RETURNED])} not returned), total {inr(neg.CONTRIBUTION_INR.sum())}.")
    table(neg[["TRANSACTION_ID", "TRANSACTION_DATE", "PRODUCT_NAME", "REGION", "SALES_CHANNEL", "DISCOUNT_PCT", "RETURNED", "NET_REVENUE_INR", "CONTRIBUTION_INR"]].head(200))


def scenario_inputs(prefix: str, defaults: S.Adjustments) -> S.Adjustments:
    c = st.columns(7)
    return S.Adjustments(
        volume_pct=c[0].number_input("Volume %", -90.0, 200.0, float(defaults.volume_pct), 1.0, key=prefix + "v"),
        price_pct=c[1].number_input("Price %", -50.0, 100.0, float(defaults.price_pct), 0.5, key=prefix + "p"),
        discount_pp=c[2].number_input("Discount pp", -30.0, 30.0, float(defaults.discount_pp), 0.5, key=prefix + "d"),
        return_pp=c[3].number_input("Return pp", -20.0, 30.0, float(defaults.return_pp), 0.5, key=prefix + "r"),
        cogs_pct=c[4].number_input("COGS %", -50.0, 100.0, float(defaults.cogs_pct), 0.5, key=prefix + "c"),
        shipping_pct=c[5].number_input("Shipping %", -100.0, 200.0, float(defaults.shipping_pct), 1.0, key=prefix + "s"),
        marketing_pct=c[6].number_input("Marketing %", -100.0, 300.0, float(defaults.marketing_pct), 1.0, key=prefix + "m"),
    )


def page_forecast(f):
    st.title("Future Profit Forecast")
    st.warning("**Estimates only.** Forecasts are built from historical (synthetic sample) data and are not guaranteed future profits or real-world predictions.")
    hist = agg(f, ("SALE_MONTH",))
    probs = F.validate_history(hist)
    if probs:
        st.error("Cannot forecast: " + " ".join(probs))
        return
    c1, c2, c3 = st.columns(3)
    horizon = c1.radio("Horizon (months)", F.HORIZONS, index=1, horizontal=True)
    use_ml = c3.toggle("Include Snowflake ML FORECAST", value=True, help="Trains a temporary SNOWFLAKE.ML.FORECAST model (uses warehouse credits)")
    bt = backtest(f, use_ml)
    methods = list(bt.method) if len(bt) else ["linear_trend"]
    method = c2.selectbox("Method", methods, format_func=lambda m: F.METHODS[m] + (" (best on holdout)" if m == methods[0] else ""))
    with st.expander("Scenario assumptions (editable)", expanded=False):
        scen = {}
        for name, d in S.SCENARIO_PRESETS.items():
            st.markdown(f"**{name}**")
            scen[name] = scenario_inputs("fc_" + name, d)
    res = run_forecast(f, horizon, method, json.dumps({k: S.to_dict(v) for k, v in scen.items()}))
    if res["fallback"]:
        st.info("Selected method unavailable - fell back to linear trend.")
    fc = res["table"]
    st.session_state["last_forecast"] = {"table": fc, "method": F.METHODS[res["method_used"]], "horizon": horizon, "filters": f}

    fig = go.Figure()
    fig.add_scatter(x=hist.SALE_MONTH, y=hist.NET_REVENUE_INR, name="Actual net revenue", mode="lines+markers")
    fig.add_scatter(x=hist.SALE_MONTH, y=hist.CONTRIBUTION_INR, name="Actual contribution", mode="lines+markers")
    fig.add_scatter(x=fc.SALE_MONTH, y=fc.UPPER_95_INR, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc.SALE_MONTH, y=fc.LOWER_95_INR, fill="tonexty", line=dict(width=0), name="95% interval (revenue)", fillcolor="rgba(255,127,14,0.15)")
    fig.add_scatter(x=fc.SALE_MONTH, y=fc.FORECAST_NET_REVENUE_INR, name="Forecast net revenue", line=dict(dash="dash"))
    for s, dash in [("CONSERVATIVE", "dot"), ("BASE_CASE", "dash"), ("OPTIMISTIC", "dot")]:
        fig.add_scatter(x=fc.SALE_MONTH, y=fc[f"{s}_CONTRIBUTION_INR"], name=f"Forecast contribution - {s.title().replace('_', ' ')}", line=dict(dash=dash))
    fig.update_layout(height=450, title="Actual vs forecast (₹)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    c = st.columns(3)
    for i, s in enumerate(["CONSERVATIVE", "BASE_CASE", "OPTIMISTIC"]):
        c[i].metric(f"{s.title().replace('_', ' ')} - {horizon}-mo contribution (est.)", inr(fc[f"{s}_CONTRIBUTION_INR"].sum()),
                    f"Net revenue {inr(fc[f'{s}_NET_REVENUE_INR'].sum())}", delta_color="off")
    table(fc)
    st.download_button("Download forecast CSV", fc.to_csv(index=False).encode(), f"forecast_{horizon}m.csv", "text/csv")

    st.subheader("Holdout evaluation (last 3 months held out)")
    if len(bt):
        table(bt.rename(columns={"MAE_INR": "MAE_INR", "MAPE_PCT": "MAPE_PCT"}))
        st.caption("Lower MAE/MAPE is better. Coverage = % of holdout months inside the 95% interval. With 9 training months these metrics are themselves uncertain.")
    else:
        st.caption("Not enough history for a holdout evaluation.")
    d = res["drivers"]
    st.subheader("Method, drivers and limitations")
    st.markdown(f"""
- **Revenue model:** {F.METHODS[res['method_used']]} on {res['history_months']} months of net revenue.
- **Contribution** is not a % of revenue: each forecast month's revenue is converted into units and transactions using the trailing
  {res['driver_window']}-month unit economics (list price ₹{d.list_price:,.0f}/unit, discount {d.discount_rate:.1%}, return value {d.return_rate:.1%},
  COGS ₹{d.cogs_per_unit:,.0f}/unit, shipping ₹{d.shipping_per_txn:,.0f}/txn, marketing ₹{d.marketing_per_txn:,.0f}/txn); scenario assumptions then adjust those drivers.
- **Limitations:** only {res['history_months']} months - seasonality (e.g. the December peak) cannot be separated from trend; no external drivers (prices, competitors, macro).
""")


def page_whatif(f):
    st.title("What-if Scenario Simulator")
    st.caption("Simulations use filtered aggregates from Snowflake and never modify the sales table.")
    k = agg(f).iloc[0].to_dict()
    if not k.get("TXN_COUNT"):
        st.info("No data for the filters.")
        return
    d = S.Drivers.from_totals(k)
    preset = st.selectbox("Start from preset", ["Custom"] + list(S.SCENARIO_PRESETS))
    a = scenario_inputs("wi_" + preset, S.SCENARIO_PRESETS.get(preset, S.Adjustments()))
    cmp = S.compare(d, a)
    st.session_state["last_scenario"] = {"table": cmp, "name": preset, "filters": f}
    base, prop = S.compute(d), S.compute(d, a)
    c = st.columns(3)
    c[0].metric("Net revenue", inr(prop["net_revenue"]), inr(prop["net_revenue"] - base["net_revenue"]))
    c[1].metric("Est. contribution", inr(prop["contribution"]), inr(prop["contribution"] - base["contribution"]))
    c[2].metric("Margin", pct(prop["margin_pct"]), f"{(prop['margin_pct'] or 0) - (base['margin_pct'] or 0):+.1f} pp")
    for w in prop["warnings"]:
        st.warning(w)
    wf = cmp[cmp.Metric.isin(["Gross sales", "Discounts", "Returns", "COGS", "Shipping", "Marketing"])].copy()
    sign = {"Gross sales": 1, "Discounts": -1, "Returns": -1, "COGS": -1, "Shipping": -1, "Marketing": -1}
    wf["Effect on contribution"] = wf.apply(lambda r: sign[r.Metric] * r.Difference, axis=1)
    fig = go.Figure(go.Waterfall(x=["Baseline"] + list(wf.Metric) + ["Scenario"],
                                 measure=["absolute"] + ["relative"] * len(wf) + ["total"],
                                 y=[base["contribution"]] + list(wf["Effect on contribution"]) + [0]))
    fig.update_layout(title="Contribution bridge: baseline → scenario (₹)", height=380)
    st.plotly_chart(fig, use_container_width=True)
    table(cmp)
    be = S.break_even(d, a)
    st.subheader("Break-even conditions (contribution = 0, other assumptions held)")
    st.markdown("\n".join([
        f"- Maximum discount rate: **{be['max_discount_rate_pct']:.1f}%** (current {d.discount_rate:.1%})" if "max_discount_rate_pct" in be else "",
        f"- Maximum return rate (by value): **{be['max_return_rate_pct']:.1f}%** (current {d.return_rate:.1%})" if "max_return_rate_pct" in be else "",
        f"- List price change to break even: **{be['price_change_to_break_even_pct']:+.1f}%**" if be.get("price_change_to_break_even_pct") is not None else "",
        f"- Maximum COGS increase: **{be['max_cogs_increase_pct']:+.1f}%**" if "max_cogs_increase_pct" in be else "",
    ]))
    st.info("Assumptions: volume changes scale units and transactions equally; no demand response to price/discount changes is modelled "
            "(a price rise is assumed to keep volume unless you also change Volume %). Returned items keep their COGS, shipping and marketing cost.")


def page_ai(f):
    st.title("AI Business Insights")
    iss = issues_df(f)
    st.subheader("Ranked findings")
    st.caption("Generated by documented rules over Snowflake query results. Score = impact (normalised) × evidence weight ÷ effort. "
               "Evidence is fact; explanations are hypotheses.")
    if empty_guard(iss, "No issues detected for these filters."):
        pass
    else:
        for r in iss.itertuples():
            with st.expander(f"#{r.rank} · {r.issue} — {r.dimension}: {r.affected} · impact {inr(r.impact_inr)} · score {r.score}"):
                st.markdown(f"**Evidence (from data):** {r.evidence}")
                st.markdown(f"**Likely explanation (unproven):** {r.explanation}")
                st.markdown(f"**Recommended action:** {r.action}")
                st.markdown(f"**Estimated impact:** {inr(r.impact_inr)} — _{r.impact_basis}_")
                st.markdown(f"**Effort:** {r.effort} · **Uncertainty:** {r.uncertainty} (n = {r.evidence_n}) · "
                            f"**Follow-up KPI:** {r.kpi} · **Measure over:** {r.period}")
        st.download_button("Download findings CSV", iss.to_csv(index=False).encode(), "findings.csv", "text/csv")

    st.subheader("Ask a question")
    examples = ["Which products contribute the most profit?", "Which categories have the highest return rate?",
                "What is reducing our contribution margin?", "Which region should management investigate?",
                "What would happen if discounts decreased by 5 percentage points?", "What are the estimated contributions for the next six months?"]
    ex = st.selectbox("Examples", [""] + examples)
    q = st.text_input("Your question", value=ex, max_chars=500)
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Interpreting question and querying Snowflake…"):
            try:
                res = AI.answer(session(), q, f)
            except Exception as e:
                st.error(f"Could not answer: {e}")
                return
        if res["summary"]:
            st.markdown(res["summary"])
            st.caption("AI-written summary of the table below (Cortex AI_COMPLETE). Verify against the numbers.")
        st.markdown("**Supporting results**")
        table(res["table"])
        if res["calc_note"]:
            st.caption(res["calc_note"])
        with st.expander("How this was answered"):
            st.json({"parsed_intent": res["intent"], "parser": res["parser"], "query": res["source"],
                     "filters": f.describe(), "safety": "LLM output is only an allow-listed intent; no LLM-generated SQL is executed."})


def page_reports(f):
    st.title("Report Generator")
    o = options()
    with st.form("report"):
        c1, c2, c3 = st.columns(3)
        dr = c1.date_input("Report period", (f.start or o["min_date"], f.end or o["max_date"]), min_value=o["min_date"], max_value=o["max_date"])
        rtype = c2.selectbox("Report type", RG.REPORT_TYPES)
        horizon = c3.selectbox("Forecast horizon", F.HORIZONS, index=1)
        c1, c2, c3, c4 = st.columns(4)
        reg = c1.multiselect("Region", o["region"], default=f.region)
        cat = c2.multiselect("Category", o["category"], default=f.category)
        ch = c3.multiselect("Sales channel", o["sales_channel"], default=f.sales_channel)
        seg = c4.multiselect("Customer segment", o["customer_segment"], default=f.customer_segment)
        scen_name = st.selectbox("Scenario", list(S.SCENARIO_PRESETS) + (["Last What-if scenario"] if "last_scenario" in st.session_state else []))
        go_ = st.form_submit_button("Preview report", type="primary")
    if go_:
        start, end = dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (o["min_date"], o["max_date"])
        rf = A.Filters(start=start, end=end, region=reg, category=cat, sales_channel=ch, customer_segment=seg, promotion_type=f.promotion_type)
        st.session_state["report_filters"] = (rf, rtype, horizon, scen_name)
    if "report_filters" not in st.session_state:
        st.info("Choose options and click **Preview report**.")
        return
    rf, rtype, horizon, scen_name = st.session_state["report_filters"]
    k = agg(rf).iloc[0].to_dict()
    if not k.get("TXN_COUNT"):
        st.warning("No transactions for these report filters.")
        return
    with st.spinner("Collecting report data from Snowflake…"):
        monthly = agg(rf, ("SALE_MONTH",))
        d = S.Drivers.from_totals(k)
        if scen_name == "Last What-if scenario":
            scen_tbl = st.session_state["last_scenario"]["table"]
        else:
            scen_tbl = S.compare(d, S.SCENARIO_PRESETS[scen_name])
        fc, meta = None, {}
        if not F.validate_history(monthly):
            res = run_forecast(rf, horizon, "linear_trend", json.dumps({k_: S.to_dict(v) for k_, v in S.SCENARIO_PRESETS.items()}))
            fc = res["table"]
            bt = F.evaluate(monthly, None, holdout=3, use_ml=False)
            acc = bt[bt.method == "linear_trend"]
            meta = {"method": F.METHODS[res["method_used"]], "history_months": res["history_months"],
                    "accuracy": (f"Holdout (last 3 months) MAPE {acc.MAPE_PCT.iloc[0]:.1f}%, MAE {RG.inr(acc.MAE_INR.iloc[0])}." if len(acc) else "")}
        ctx = dict(
            title="Retail Sales Management Report", report_type=rtype, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            filters=rf.describe(), coverage={"Transactions in scope": f"{int(k['TXN_COUNT']):,}",
                                             "Months in scope": str(len(monthly)), "Source": "RETAIL_ANALYTICS.SALES.V_SALES_OVERVIEW"},
            kpis=k, monthly=monthly, category=agg(rf, ("CATEGORY",)), products=agg(rf, ("PRODUCT_NAME",)),
            region=agg(rf, ("REGION",)), channel=agg(rf, ("SALES_CHANNEL",)), issues=issues_df(rf),
            forecast=fc if rtype in ("Executive", "Forecast") else None, forecast_meta=meta,
            scenario=scen_tbl, scenario_name=scen_name, dq=dq_latest(), assumptions=RG.DEFAULT_ASSUMPTIONS)
    st.subheader("Preview")
    st.json(rf.describe())
    kpi_row(k)
    st.markdown("**Top findings**")
    table(ctx["issues"][["rank", "issue", "affected", "impact_inr"]].head(5))
    if fc is not None:
        st.markdown(f"**Forecast ({horizon} months, estimate)**")
        table(fc)
    pdf = RG.build_pdf(ctx)
    tx = A.filtered_transactions(session(), rf)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Download PDF report", pdf, f"retail_{rtype.lower()}_report_{stamp}.pdf", "application/pdf", type="primary")
    c2.download_button(f"Filtered transactions CSV ({len(tx):,} rows)", tx.to_csv(index=False).encode(), f"filtered_transactions_{stamp}.csv", "text/csv")
    if fc is not None:
        c3.download_button("Forecast CSV", fc.to_csv(index=False).encode(), f"forecast_{horizon}m_{stamp}.csv", "text/csv")
    c4.download_button("Data-quality summary CSV", ctx["dq"].to_csv(index=False).encode(), f"data_quality_{stamp}.csv", "text/csv")
    ok = abs(tx.NET_REVENUE_INR.sum() - k["NET_REVENUE_INR"]) < 0.01 and len(tx) == k["TXN_COUNT"]
    (st.success if ok else st.error)(f"Export check: CSV rows {len(tx):,} / net revenue {inr(tx.NET_REVENUE_INR.sum(), 2)} "
                                     f"{'match' if ok else 'DO NOT match'} report KPIs. PDF size {len(pdf) / 1024:.0f} KB.")


def page_upload(_f):
    st.title("Upload Data")
    st.caption("CSV → preview → column mapping → validation → (optional) corrections → load. Nothing is dropped silently.")
    up = st.file_uploader("CSV file", type=["csv"])
    if not up:
        st.info("Upload the sample `retail_sales_analytics_sample.csv` or your own CSV with similar columns.")
        hist = A.upload_history(session())
        if len(hist):
            st.subheader("Upload history")
            table(hist)
        return
    raw_bytes = up.getvalue()
    if st.session_state.get("upload_name") != up.name + str(len(raw_bytes)):
        try:
            st.session_state["raw_df"] = V.read_csv(raw_bytes)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            return
        st.session_state["upload_name"] = up.name + str(len(raw_bytes))
    raw = st.session_state["raw_df"]
    st.subheader(f"Preview - {len(raw):,} rows, {len(raw.columns)} columns")
    st.dataframe(raw.head(20), use_container_width=True)

    st.subheader("Column mapping")
    sug = V.suggest_mapping(list(raw.columns))
    mapping = {}
    cols = st.columns(4)
    opts = ["(not in file)"] + list(raw.columns)
    for i, req in enumerate(V.REQUIRED_COLUMNS):
        default = opts.index(sug[req]) if sug[req] in opts else 0
        sel = cols[i % 4].selectbox(req + (" *" if req in V.CRITICAL_COLUMNS else ""), opts, index=default, key="map_" + req)
        mapping[req] = None if sel == "(not in file)" else sel
    st.caption("* required. Optional columns that are absent are derived where a formula exists; missing cost columns make contribution unavailable.")

    target_mode = st.radio("Load mode", ["APPEND", "REPLACE", "NEW_TABLE"], horizontal=True,
                           captions=["Add rows to SALES_TRANSACTIONS", "Delete existing rows then load (needs confirmation)", "Create a separate table"])
    target = "SALES_TRANSACTIONS" if target_mode != "NEW_TABLE" else st.text_input("New table name", "SALES_TRANSACTIONS_UPLOAD")

    mapped = V.apply_mapping(raw, mapping)
    typed, notes = V.coerce_and_derive(mapped, mapping)
    try:
        ids = L.existing_ids(session(), target) if target_mode == "APPEND" else set()
    except Exception:
        ids = set()
    res = V.validate(typed, mapping, ids)
    s, fr = res["summary"], res["frame"]
    st.subheader("Validation report")
    c = st.columns(4)
    c[0].metric("Rows in file", f"{s['rows_in_file']:,}")
    c[1].metric("Valid rows", f"{s['valid_rows']:,}")
    c[2].metric("Invalid rows", f"{s['invalid_rows']:,}")
    c[3].metric("Rows with warnings", f"{s['rows_with_warnings']:,}")
    for n in notes:
        st.info(n)
    if s["issue_counts"]:
        table(pd.DataFrame([{"Issue": k_, "Rows": v} for k_, v in sorted(s["issue_counts"].items(), key=lambda x: -x[1])]))
    if res["blocking"]:
        st.error(f"Required columns not mapped: {', '.join(s['unmapped_critical_columns'])}. Map them to continue.")
        return
    bad = fr[~fr._valid | (fr._warnings != "")]
    if len(bad):
        st.subheader("Affected rows")
        st.dataframe(bad[["_source_row", "transaction_id", "_errors", "_warnings"]], hide_index=True, use_container_width=True)
        inv = fr[~fr._valid]
        if len(inv):
            with st.expander(f"Correct {len(inv)} invalid row(s) in place"):
                src_cols = [mapping[c_] for c_ in V.REQUIRED_COLUMNS if mapping.get(c_)]
                edited = st.data_editor(raw.loc[inv.index, src_cols], key="fix_editor", use_container_width=True)
                if st.button("Apply corrections and re-validate"):
                    st.session_state["raw_df"].loc[edited.index, src_cols] = edited.astype(str)
                    st.rerun()

    choice = st.radio("Invalid rows", ["Load valid rows only (invalid rows recorded as rejected)", "Reject the whole file"],
                      disabled=s["invalid_rows"] == 0) if s["invalid_rows"] else "Load all rows"
    confirm = True
    if target_mode == "REPLACE":
        st.warning("REPLACE deletes **all** existing rows in SALES_TRANSACTIONS before loading.")
        confirm = st.text_input("Type REPLACE to confirm") == "REPLACE"
    if st.button("Load into Snowflake", type="primary", disabled=not confirm):
        if choice.startswith("Reject"):
            L.record_rejection(session(), up.name, s["rows_in_file"], s)
            st.warning("File rejected and recorded in UPLOAD_HISTORY. No data loaded.")
            return
        good = fr[fr._valid]
        if len(good) == 0:
            st.error("No valid rows to load.")
            return
        with st.spinner("Staging file and loading rows…"):
            try:
                out = L.load(session(), V.clean_for_load(good), file_name=up.name, raw_bytes=raw_bytes, mode=target_mode, target_table=target,
                             rows_in_file=s["rows_in_file"], rows_rejected=s["invalid_rows"],
                             summary={k_: v for k_, v in s.items()} | {"derived": notes}, confirm_replace=confirm)
                if out["target"] == "SALES_TRANSACTIONS":
                    A.run_data_quality(session())
            except Exception as e:
                st.error(f"Load failed (rolled back, recorded in UPLOAD_HISTORY): {e}")
                return
        st.cache_data.clear()
        st.success(f"Loaded {out['rows_loaded']:,} rows into {out['target']} (status {out['status']}, upload id {out['upload_id']}).")
        if out["target"] != "SALES_TRANSACTIONS":
            st.info("Dashboards read SALES_TRANSACTIONS; the new table is stored separately for review.")


def page_quality(_f):
    st.title("Data Quality")
    if st.button("Run checks now"):
        with st.spinner("Running RUN_DATA_QUALITY_CHECKS()…"):
            A.run_data_quality(session())
        st.cache_data.clear()
    dq = dq_latest()
    if empty_guard(dq, "No checks have been run yet."):
        return
    st.caption(f"Last run: {pd.to_datetime(dq.CHECKED_AT.iloc[0]):%Y-%m-%d %H:%M}")
    table(dq.drop(columns=["CHECKED_AT"]))
    st.subheader("Reconciliation: base table vs analytic views")
    rc = A.reconcile(session())
    table(rc)
    ok = rc[["TXN_COUNT", "NET_REVENUE_INR", "CONTRIBUTION_INR"]].nunique().max() == 1
    (st.success if ok else st.error)("All sources reconcile." if ok else "Mismatch between sources - investigate before using the dashboards.")
    st.download_button("Download data-quality CSV", dq.to_csv(index=False).encode(), "data_quality.csv", "text/csv")


def page_about(_f):
    st.title("Methodology & Project Info")
    st.markdown("""
**Data:** `RETAIL_ANALYTICS.SALES.SALES_TRANSACTIONS` (INR). The bundled sample is **synthetic** - costs and margins are simulated.

| Measure | Formula |
|---|---|
| Gross sales | Σ unit_price_inr × quantity |
| Discount | Σ discount_amount_inr |
| Revenue before returns | Σ revenue_inr (= gross − discount) |
| Net revenue | Σ net_revenue_inr (= revenue − return_amount) |
| COGS | Σ cost_of_goods_inr |
| **Estimated contribution** | Σ (net revenue − COGS − shipping − marketing) |
| Contribution margin | contribution ÷ net revenue × 100 (n/a when net revenue = 0) |
| Average order value | net revenue ÷ distinct transaction IDs |
| Return rate | returned transactions ÷ distinct transactions × 100 |

**Returns:** a return reverses the full revenue of the transaction; COGS, shipping and marketing stay charged (no inventory recovery or
extra handling cost is in the data). **Discounts** are counted once (already removed in revenue). **Contribution is not full accounting profit**
- other operating expenses are not provided. Duplicate transaction IDs are flagged by data-quality checks; counts use distinct IDs.

**AI:** findings are rule-based over query results; Cortex AI_COMPLETE only (a) turns a question into an allow-listed intent and
(b) summarises the resulting table. No LLM-generated SQL is executed.

**Forecasting:** recent-mean baseline, linear trend (with 95% prediction interval) and Snowflake ML FORECAST, compared on a 3-month holdout.
Contribution forecasts come from driver-based unit economics, and scenarios change those drivers.
""")
    st.dataframe(pd.DataFrame([{"Assumption": a} for a in RG.DEFAULT_ASSUMPTIONS]), hide_index=True, use_container_width=True)


PAGES = {
    "Analytics": [("Executive Overview", page_overview), ("Products & Categories", page_products),
                  ("Regions & Channels", page_regions), ("Discounts, Returns & Costs", page_costs)],
    "Planning": [("Profit Forecast", page_forecast), ("What-if Simulator", page_whatif), ("AI Insights", page_ai)],
    "Data & Reports": [("Report Generator", page_reports), ("Upload Data", page_upload), ("Data Quality", page_quality),
                       ("Methodology", page_about)],
}


def main():
    st.sidebar.title("Retail Analytics")
    flat = [(g, n, fn) for g, items in PAGES.items() for n, fn in items]
    choice = st.sidebar.radio("Navigate", [n for _, n, _ in flat], format_func=lambda n: n)
    f = sidebar_filters()
    fn = next(fn for _, n, fn in flat if n == choice)
    try:
        fn(f)
    except Exception as e:  # user-friendly error, details in expander
        st.error("Something went wrong while loading this page. Try Refresh data or widen the filters.")
        with st.expander("Technical details"):
            st.exception(e)


main()
