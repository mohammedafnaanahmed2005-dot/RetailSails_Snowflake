"""PDF management report (ReportLab only - available in the Snowflake Anaconda channel).

build_pdf() is a pure function of already-queried data, so it can be unit tested
and guarantees the PDF shows exactly the numbers on screen for the same filters.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

BLUE, ORANGE, GREEN, GREY = colors.HexColor("#1f77b4"), colors.HexColor("#ff7f0e"), colors.HexColor("#2ca02c"), colors.HexColor("#7f7f7f")
REPORT_TYPES = ("Executive", "Profitability", "Operations", "Forecast")


def inr(v, dec=0) -> str:
    """Indian-style INR formatting with lakh/crore grouping, safe for None/NaN."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    neg, v = v < 0, abs(float(v))
    s = f"{v:.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        whole = head + "," + tail
    return ("-" if neg else "") + "Rs " + whole + ("." + frac if frac else "")


def pct(v, dec=1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{dec}f}%"


_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Title"], fontSize=18, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], textColor=colors.HexColor("#11567F"), spaceBefore=10)
BODY = ParagraphStyle("b", parent=_styles["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.5, leading=9.5, textColor=GREY)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.5, leading=9)


def _table(df: pd.DataFrame, fmt: dict | None = None, widths=None, max_rows=20) -> Table:
    fmt = fmt or {}
    df = df.head(max_rows)
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(str(fmt[c](r[c]) if c in fmt else ("" if pd.isna(r[c]) else r[c])), CELL) for c in df.columns])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F4FB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    return t


def _line_chart(series: list[tuple[str, list[float], colors.Color]], labels: list[str], title: str, w=17 * cm, h=6.5 * cm) -> Drawing:
    d = Drawing(w, h)
    d.add(String(0, h - 12, title, fontSize=9, fillColor=colors.black))
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 45, 30, w - 150, h - 55
    lp.data = [[(i, float(v)) for i, v in enumerate(vals) if v is not None and not pd.isna(v)] for _, vals, _ in series]
    for i, (_, _, c) in enumerate(series):
        lp.lines[i].strokeColor = c
        lp.lines[i].strokeWidth = 1.5
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 0, max(len(labels) - 1, 1)
    step = max(1, len(labels) // 8)
    lp.xValueAxis.valueSteps = list(range(0, len(labels), step))
    lp.xValueAxis.labelTextFormat = lambda v: labels[int(v)] if 0 <= int(v) < len(labels) else ""
    lp.xValueAxis.labels.fontSize = 6.5
    lp.yValueAxis.labels.fontSize = 6.5
    lp.yValueAxis.labelTextFormat = lambda v: f"{v / 1e5:.1f}L"
    d.add(lp)
    lg = Legend()
    lg.x, lg.y = w - 100, h - 30
    lg.fontSize = 7
    lg.colorNamePairs = [(c, n) for n, _, c in series]
    d.add(lg)
    return d


def _bar_chart(names: list[str], vals: list[float], title: str, w=17 * cm, h=6 * cm) -> Drawing:
    d = Drawing(w, h)
    d.add(String(0, h - 12, title, fontSize=9))
    bc = HorizontalBarChart()
    bc.x, bc.y, bc.width, bc.height = 110, 15, w - 140, h - 35
    bc.data = [[float(v) for v in vals]]
    bc.categoryAxis.categoryNames = [n[:22] for n in names]
    bc.categoryAxis.labels.fontSize = 6.5
    bc.valueAxis.labels.fontSize = 6.5
    bc.valueAxis.labelTextFormat = lambda v: f"{v / 1e5:.1f}L"
    bc.bars[0].fillColor = BLUE
    d.add(bc)
    return d


def build_pdf(ctx: dict) -> bytes:
    """ctx keys: title, report_type, filters(dict), coverage(dict), kpis(dict), monthly(df), category(df),
    products(df), region(df), channel(df), issues(df), forecast(df|None), forecast_meta(dict),
    scenario(df|None), scenario_name, dq(df), assumptions(list[str])"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title=ctx.get("title", "Retail report"), author="Retail Sales Analytics")
    rt = ctx.get("report_type", "Executive")
    k = ctx["kpis"]
    el = []
    gen = ctx.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")

    # 1-2 Title, filters, coverage
    el += [Paragraph(ctx.get("title", "Retail Sales Management Report"), H1),
           Paragraph(f"{rt} report &middot; generated {gen} &middot; all amounts in INR (Rs)", SMALL), Spacer(1, 6),
           Paragraph("Selected filters and data coverage", H2),
           _table(pd.DataFrame([{"Filter": a, "Value": b} for a, b in {**ctx["filters"], **ctx.get("coverage", {})}.items()]),
                  widths=[5 * cm, 12 * cm])]

    # 3 Executive summary
    issues = ctx.get("issues", pd.DataFrame())
    top = issues.head(3) if len(issues) else issues
    summ = (f"Net revenue was <b>{inr(k.get('NET_REVENUE_INR'))}</b> across {int(k.get('TXN_COUNT') or 0):,} transactions, "
            f"with an estimated contribution of <b>{inr(k.get('CONTRIBUTION_INR'))}</b> ({pct(k.get('MARGIN_PCT'))} margin). "
            f"Average order value was {inr(k.get('AOV_INR'))} and the return rate {pct(k.get('RETURN_RATE_PCT'))}. ")
    if len(top):
        summ += "Top-ranked issues: " + "; ".join(f"{r.issue} ({r.affected})" for r in top.itertuples()) + "."
    el += [Paragraph("Executive summary", H2), Paragraph(summ, BODY),
           Paragraph("Contribution = net revenue - COGS - shipping - marketing. It is an estimate before other unprovided "
                     "expenses and is not full accounting profit.", SMALL)]

    # 4 KPIs
    kp = [("Net revenue", inr(k.get("NET_REVENUE_INR"))), ("Estimated contribution", inr(k.get("CONTRIBUTION_INR"))),
          ("Contribution margin", pct(k.get("MARGIN_PCT"))), ("Transactions", f"{int(k.get('TXN_COUNT') or 0):,}"),
          ("Average order value", inr(k.get("AOV_INR"))), ("Return rate", pct(k.get("RETURN_RATE_PCT"))),
          ("Gross sales", inr(k.get("GROSS_SALES_INR"))), ("Discounts", inr(k.get("DISCOUNT_INR"))),
          ("Returns (value)", inr(k.get("RETURNS_INR")))]
    el += [Paragraph("Key performance indicators (actual)", H2),
           _table(pd.DataFrame([[a, b, c, d] for (a, b), (c, d) in zip(kp[0::2], kp[1::2] + [("", "")])],
                               columns=["KPI", "Value", "KPI ", "Value "]), widths=[4.5 * cm, 4 * cm, 4.5 * cm, 4 * cm])]

    # 5 Monthly trend
    m = ctx.get("monthly", pd.DataFrame())
    if len(m):
        labels = [str(pd.to_datetime(x).date())[:7] for x in m.SALE_MONTH]
        el += [Paragraph("Monthly trend", H2),
               _line_chart([("Net revenue", list(m.NET_REVENUE_INR), BLUE), ("Contribution", list(m.CONTRIBUTION_INR), ORANGE)],
                           labels, "Net revenue and estimated contribution by month (Rs lakh)")]

    # 6 Product & category
    if rt in ("Executive", "Profitability", "Operations"):
        c = ctx.get("category", pd.DataFrame())
        p = ctx.get("products", pd.DataFrame())
        money = {x: inr for x in ["NET_REVENUE_INR", "CONTRIBUTION_INR", "AOV_INR", "AVG_SELLING_PRICE_INR", "RETURNS_INR"]}
        pcts = {x: pct for x in ["MARGIN_PCT", "RETURN_RATE_PCT", "EFFECTIVE_DISCOUNT_PCT"]}
        if len(c):
            el += [Paragraph("Category performance", H2),
                   _bar_chart(list(c.CATEGORY), list(c.CONTRIBUTION_INR), "Estimated contribution by category (Rs lakh)"),
                   _table(c[["CATEGORY", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "RETURN_RATE_PCT", "TXN_COUNT"]], {**money, **pcts})]
        if len(p):
            cols = ["PRODUCT_NAME", "UNITS", "AVG_SELLING_PRICE_INR", "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "RETURN_RATE_PCT"]
            el += [Paragraph("Product performance (by contribution)", H2),
                   _table(p.sort_values("CONTRIBUTION_INR", ascending=False)[cols], {**money, **pcts, "UNITS": lambda v: f"{int(v):,}"})]
        # 7 Region & channel
        for key, col, title in [("region", "REGION", "Regional analysis"), ("channel", "SALES_CHANNEL", "Channel analysis")]:
            df = ctx.get(key, pd.DataFrame())
            if len(df):
                el += [Paragraph(title, H2),
                       _table(df[[col, "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "AOV_INR", "RETURN_RATE_PCT",
                                  "EFFECTIVE_DISCOUNT_PCT", "TXN_COUNT"]], {**money, **pcts})]

    # 8-9 Problems and actions
    if len(issues):
        el += [PageBreak(), Paragraph("Major business problems and recommended actions", H2),
               Paragraph("Ranked by estimated impact, weighted by evidence (sample size) and implementation effort. "
                         "Evidence = facts from the data; explanations are hypotheses to be tested.", SMALL)]
        for r in issues.head(10).itertuples():
            el.append(KeepTogether([
                Paragraph(f"<b>{r.rank}. {r.issue} - {r.dimension}: {r.affected}</b>", BODY),
                Paragraph(f"<b>Evidence:</b> {r.evidence}", CELL),
                Paragraph(f"<b>Likely explanation (unproven):</b> {r.explanation}", CELL),
                Paragraph(f"<b>Action:</b> {r.action}", CELL),
                Paragraph(f"<b>Estimated impact:</b> {inr(r.impact_inr)} &middot; {r.impact_basis}", CELL),
                Paragraph(f"<b>Effort:</b> {r.effort} &middot; <b>Uncertainty:</b> {r.uncertainty} &middot; "
                          f"<b>KPI:</b> {r.kpi} &middot; <b>Measure over:</b> {r.period}", CELL),
                Spacer(1, 5)]))

    # 10-11 Forecast & scenarios
    fc = ctx.get("forecast")
    if fc is not None and len(fc):
        meta = ctx.get("forecast_meta", {})
        el += [Paragraph(f"Forecast - next {len(fc)} months (ESTIMATE, not guaranteed)", H2),
               Paragraph(f"Method: {meta.get('method', '')}. History: {meta.get('history_months', '?')} months. "
                         f"{meta.get('accuracy', '')}", SMALL)]
        hist = ctx.get("monthly", pd.DataFrame())
        labels = [str(pd.to_datetime(x).date())[:7] for x in list(hist.SALE_MONTH) + list(fc.SALE_MONTH)]
        nh = len(hist)
        el.append(_line_chart([
            ("Actual net revenue", list(hist.NET_REVENUE_INR) + [None] * len(fc), BLUE),
            ("Forecast net revenue", [None] * (nh - 1) + [list(hist.NET_REVENUE_INR)[-1]] + list(fc.FORECAST_NET_REVENUE_INR), ORANGE),
            ("Base-case contribution", [None] * nh + list(fc.BASE_CASE_CONTRIBUTION_INR), GREEN)],
            labels, "Actual vs forecast (Rs lakh)"))
        show = fc[["SALE_MONTH", "FORECAST_NET_REVENUE_INR", "LOWER_95_INR", "UPPER_95_INR", "CONSERVATIVE_CONTRIBUTION_INR",
                   "BASE_CASE_CONTRIBUTION_INR", "OPTIMISTIC_CONTRIBUTION_INR"]].copy()
        show.columns = ["Month", "Net revenue (fc)", "Lower 95%", "Upper 95%", "Contrib. conservative", "Contrib. base", "Contrib. optimistic"]
        el.append(_table(show, {c: inr for c in show.columns if c != "Month"}))
        tot = pd.DataFrame([{"Scenario": s, "Net revenue": inr(fc[f"{s.upper().replace(' ', '_')}_NET_REVENUE_INR"].sum()),
                             "Contribution": inr(fc[f"{s.upper().replace(' ', '_')}_CONTRIBUTION_INR"].sum())}
                            for s in ["Conservative", "Base case", "Optimistic"]])
        el += [Paragraph("Forecast horizon totals by scenario", BODY), _table(tot)]
    sc = ctx.get("scenario")
    if sc is not None and len(sc):
        el += [Paragraph(f"Scenario comparison - {ctx.get('scenario_name', 'custom')}", H2),
               _table(sc.assign(**{c: sc.apply(lambda r, c=c: pct(r[c]) if "margin" in r["Metric"].lower() else inr(r[c]), axis=1)
                                   for c in ["Baseline", "Scenario", "Difference"]}))]

    # 12-13 Methodology & DQ
    el += [Paragraph("Assumptions, methodology and limitations", H2)]
    el += [Paragraph("&bull; " + a, CELL) for a in ctx.get("assumptions", [])]
    dq = ctx.get("dq", pd.DataFrame())
    if len(dq):
        el += [Paragraph("Data-quality checks", H2),
               _table(dq[["CHECK_NAME", "SEVERITY", "STATUS", "FAILED_ROWS", "TOTAL_ROWS"]], max_rows=30)]

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(1.6 * cm, 0.8 * cm, "Retail Sales Analytics & AI Insights - estimates based on supplied data; forecasts are not guarantees.")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


DEFAULT_ASSUMPTIONS = [
    "Monetary values are INR as supplied. The sample dataset is synthetic; costs and margins are simulated.",
    "Estimated contribution = net revenue - cost of goods - shipping - marketing. Other operating expenses are not provided, so this is not full accounting profit.",
    "Returns fully reverse revenue (return amount = revenue). COGS, shipping and marketing remain charged on returned transactions: no inventory recovery or extra handling cost is recorded.",
    "Discounts are counted once: revenue = gross sales - discount amount.",
    "Return rate = returned transactions / distinct transactions. AOV = net revenue / distinct transactions.",
    "Impact estimates are upper-bound what-ifs based on closing a gap to the average; they are not causal effects.",
    "Forecasts use only historical monthly data; with 12 months of history, seasonality cannot be estimated. Treat as directional estimates.",
]
