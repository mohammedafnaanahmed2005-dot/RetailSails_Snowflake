"""Evidence-based findings and recommendations from verified query results.

Every finding is produced by a rule over Snowflake aggregates (no invented
numbers). Each carries: evidence (facts), likely explanations (hypotheses,
labelled as such), action, impact estimate + basis, effort, uncertainty,
follow-up KPI and measurement period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import analytics as A
from src.scenarios import Adjustments, Drivers, compute

EFFORT = {"pricing": "Medium", "promotion": "Low", "returns": "Medium", "region": "High",
          "transactions": "Low", "logistics": "Medium", "anomaly": "Low", "discount": "Low"}
MIN_TXNS = 30  # minimum sample size for a segment-level finding


def _uncertainty(n: int) -> str:
    return "Low" if n >= 200 else ("Medium" if n >= 80 else "High")


def _inr(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"₹{v:,.0f}"


def detect_issues(session, f: A.Filters | None) -> list[dict]:
    tot = A.kpis(session, f)
    if not tot or not tot.get("TXN_COUNT"):
        return []
    if tot.get("MISSING_COST_ROWS"):
        return [{"issue": "Contribution unavailable", "dimension": "DATA", "affected": "All",
                 "evidence": f"{int(tot['MISSING_COST_ROWS'])} rows lack cost data", "explanation": "Cost columns missing in upload",
                 "action": "Upload COGS, shipping and marketing costs", "impact_inr": None,
                 "impact_basis": "Requires cost data", "effort": "Low", "uncertainty": "High",
                 "kpi": "Rows with cost data", "period": "Next upload", "evidence_n": 0, "category": "data"}]
    margin, rr = tot["MARGIN_PCT"] or 0, tot["RETURN_RATE_PCT"] or 0
    issues: list[dict] = []

    # 1. High revenue / low margin products
    prod = A.by_dimension(session, f, "PRODUCT_NAME")
    if len(prod):
        med = prod["NET_REVENUE_INR"].median()
        for _, p in prod[(prod.NET_REVENUE_INR >= med) & (prod.MARGIN_PCT < margin - 2) & (prod.TXN_COUNT >= MIN_TXNS)].iterrows():
            disc_gap = p.EFFECTIVE_DISCOUNT_PCT - (tot["EFFECTIVE_DISCOUNT_PCT"] or 0)
            issues.append({
                "category": "pricing", "issue": "High sales but weak contribution", "dimension": "Product",
                "affected": p.PRODUCT_NAME,
                "evidence": f"Net revenue {_inr(p.NET_REVENUE_INR)} ({p.NET_REVENUE_INR / tot['NET_REVENUE_INR'] * 100:.1f}% share); "
                            f"margin {p.MARGIN_PCT:.1f}% vs overall {margin:.1f}%; COGS = {p.COGS_INR / p.NET_REVENUE_INR * 100:.1f}% of net revenue; "
                            f"effective discount {p.EFFECTIVE_DISCOUNT_PCT:.1f}% ({disc_gap:+.1f} pp vs overall); return rate {p.RETURN_RATE_PCT:.1f}%",
                "explanation": "Hypothesis: high unit cost relative to selling price" + (" and above-average discounting" if disc_gap > 0.5 else "")
                               + ". Not proven - supplier cost and price elasticity data would be needed.",
                "action": "Renegotiate supplier cost or test a 2-3% price increase; cap discounts on this product.",
                "impact_inr": p.NET_REVENUE_INR * (margin - p.MARGIN_PCT) / 100,
                "impact_basis": "Upper bound: product revenue x gap to average margin (if margin reached the average at same volume).",
                "kpi": "Product contribution margin %", "period": "Next 3 months vs same-length prior period",
                "evidence_n": int(p.TXN_COUNT)})

    # 2. High return rates by dimension
    for dim_label, col in [("Product", "PRODUCT_NAME"), ("Category", "CATEGORY"), ("Region", "REGION"), ("Sales channel", "SALES_CHANNEL")]:
        df = prod if col == "PRODUCT_NAME" else A.by_dimension(session, f, col)
        for _, r in df[(df.TXN_COUNT >= MIN_TXNS) & (df.RETURN_RATE_PCT > rr * 1.25) & (df.RETURNED_TXNS >= 3)].iterrows():
            excess_txn = (r.RETURN_RATE_PCT - rr) / 100 * r.TXN_COUNT
            avg_ret_val = r.RETURNS_INR / r.RETURNED_TXNS if r.RETURNED_TXNS else 0
            issues.append({
                "category": "returns", "issue": "High return rate", "dimension": dim_label, "affected": r[col],
                "evidence": f"Return rate {r.RETURN_RATE_PCT:.1f}% ({int(r.RETURNED_TXNS)}/{int(r.TXN_COUNT)} txns) vs overall {rr:.1f}%; "
                            f"returned revenue {_inr(r.RETURNS_INR)}",
                "explanation": "Possible causes (unverified): product quality, mismatch with description, or fulfilment damage. Return-reason data is not in the dataset.",
                "action": "Collect return reasons; review product listings, packaging and QC for this segment.",
                "impact_inr": excess_txn * avg_ret_val,
                "impact_basis": f"Revenue retained if return rate fell to the overall average: {excess_txn:.1f} fewer returns x avg returned value {_inr(avg_ret_val)}.",
                "kpi": "Return rate %", "period": "Next 3 months", "evidence_n": int(r.TXN_COUNT)})

    # 3. Promotions vs no-promotion baseline
    promo = A.by_dimension(session, f, "PROMOTION_TYPE")
    base = promo[promo.PROMOTION_TYPE == "No Promotion"]
    if len(base):
        b = base.iloc[0]
        for _, p in promo[(promo.PROMOTION_TYPE != "No Promotion") & (promo.TXN_COUNT >= MIN_TXNS)].iterrows():
            if p.MARGIN_PCT < b.MARGIN_PCT - 2:
                units_lift = (p.UNITS / p.TXN_COUNT) / (b.UNITS / b.TXN_COUNT) - 1 if b.TXN_COUNT and b.UNITS else 0
                issues.append({
                    "category": "promotion", "issue": "Promotion reducing contribution margin", "dimension": "Promotion",
                    "affected": p.PROMOTION_TYPE,
                    "evidence": f"Margin {p.MARGIN_PCT:.1f}% vs no-promotion {b.MARGIN_PCT:.1f}%; effective discount {p.EFFECTIVE_DISCOUNT_PCT:.1f}% vs {b.EFFECTIVE_DISCOUNT_PCT:.1f}%; "
                                f"units/txn {p.UNITS / p.TXN_COUNT:.2f} vs {b.UNITS / b.TXN_COUNT:.2f} ({units_lift:+.0%}); AOV {_inr(p.AOV_INR)} vs {_inr(b.AOV_INR)}",
                    "explanation": "Correlation, not causation: promoted transactions have lower margins, but product mix may differ and there is no control group to measure incremental volume.",
                    "action": "Run a controlled test with reduced discount depth; restrict promotion to higher-margin products.",
                    "impact_inr": p.NET_REVENUE_INR * (b.MARGIN_PCT - p.MARGIN_PCT) / 100,
                    "impact_basis": "Upper bound: promo revenue x margin gap vs no-promotion; ignores any volume the promotion generated.",
                    "kpi": "Promotion contribution margin % and incremental units", "period": "One promotion cycle (4-8 weeks) with control group",
                    "evidence_n": int(p.TXN_COUNT)})

    # 4. Regional gaps (volume-aware: also report AOV and mix)
    reg = A.by_dimension(session, f, "REGION")
    for _, r in reg[(reg.MARGIN_PCT < margin - 1.5) & (reg.TXN_COUNT >= MIN_TXNS)].iterrows():
        issues.append({
            "category": "region", "issue": "Regional performance gap", "dimension": "Region", "affected": r.REGION,
            "evidence": f"Margin {r.MARGIN_PCT:.1f}% vs overall {margin:.1f}% on {int(r.TXN_COUNT)} txns; return rate {r.RETURN_RATE_PCT:.1f}%; "
                        f"effective discount {r.EFFECTIVE_DISCOUNT_PCT:.1f}%; AOV {_inr(r.AOV_INR)}",
            "explanation": "May reflect product mix (e.g. more low-margin products), returns or discounting in this region - check the Product/Region breakdown before acting.",
            "action": "Drill into region x product mix; review regional discount approvals and return handling.",
            "impact_inr": r.NET_REVENUE_INR * (margin - r.MARGIN_PCT) / 100,
            "impact_basis": "Upper bound: regional revenue x gap to average margin.",
            "kpi": "Regional contribution margin %", "period": "Next quarter", "evidence_n": int(r.TXN_COUNT)})

    # 5. Discount depth vs margin (correlation only)
    bands = A.discount_bands(session, f)
    if len(bands) >= 3 and bands.TXN_COUNT.sum() >= MIN_TXNS:
        corr = float(np.corrcoef(bands.DISCOUNT_PCT, bands.MARGIN_PCT.fillna(0))[0, 1])
        hi = bands[bands.DISCOUNT_PCT >= 15]
        if corr < -0.5 and len(hi):
            d = Drivers.from_totals(tot)
            what_if = compute(d, Adjustments(discount_pp=-min(5, d.discount_rate * 100)))["contribution"] - compute(d)["contribution"]
            issues.append({
                "category": "discount", "issue": "Deep discounts associated with lower margin", "dimension": "Discount band",
                "affected": ", ".join(f"{x:.0f}%" for x in hi.DISCOUNT_PCT),
                "evidence": f"Correlation between discount % and margin across {len(bands)} bands: {corr:.2f}; "
                            f"≥15% discount bands: {int(hi.TXN_COUNT.sum())} txns, margin {hi.CONTRIBUTION_INR.sum() / hi.NET_REVENUE_INR.sum() * 100:.1f}%",
                "explanation": "Mechanical effect: discount reduces price while cost is unchanged. Whether discounts drive extra volume cannot be determined from this data.",
                "action": "Introduce a discount approval threshold above 10-15%; test smaller discounts.",
                "impact_inr": what_if,
                "impact_basis": "Scenario model: 5 pp lower discount rate at unchanged volume (no volume loss assumed - optimistic).",
                "kpi": "Effective discount % and margin %", "period": "Next 3 months", "evidence_n": int(bands.TXN_COUNT.sum())})

    # 6. Logistics cost trend
    mon = A.monthly(session, f)
    if len(mon) >= 6:
        mon["SHIP_PER_TXN"] = mon.SHIPPING_INR / mon.TXN_COUNT
        h1, h2 = mon.head(len(mon) // 2).SHIP_PER_TXN.mean(), mon.tail(len(mon) // 2).SHIP_PER_TXN.mean()
        if h1 and (h2 / h1 - 1) > 0.05:
            issues.append({
                "category": "logistics", "issue": "Rising shipping cost per transaction", "dimension": "Cost", "affected": "Shipping",
                "evidence": f"Shipping per txn {_inr(h1)} (first half) -> {_inr(h2)} (second half), {h2 / h1 - 1:+.1%}",
                "explanation": "Could be channel mix (more online orders) or carrier rates - check by channel.",
                "action": "Review carrier contracts and free-shipping thresholds.",
                "impact_inr": (h2 - h1) * mon.tail(len(mon) // 2).TXN_COUNT.sum(),
                "impact_basis": "Extra shipping spend in the second half vs first-half cost per txn.",
                "kpi": "Shipping cost per transaction", "period": "Next quarter", "evidence_n": int(mon.TXN_COUNT.sum())})
        burden = (tot["SHIPPING_INR"] + tot["MARKETING_INR"]) / tot["NET_REVENUE_INR"] * 100
        low_val = A.run(session, f"SELECT COUNT(*) N, SUM(CONTRIBUTION_INR) C FROM {A.BASE_VIEW}" +
                        (A.build_where(f)[0] + " AND " if A.build_where(f)[0] else " WHERE ") +
                        "NOT RETURNED AND CONTRIBUTION_INR < 0", A.build_where(f)[1])
        if int(low_val.N.iloc[0] or 0) > 0:
            issues.append({
                "category": "transactions", "issue": "Unprofitable non-returned transactions", "dimension": "Transaction", "affected": "Low-value / high-discount orders",
                "evidence": f"{int(low_val.N.iloc[0])} non-returned transactions with negative contribution, total {_inr(low_val.C.iloc[0])}; shipping+marketing = {burden:.1f}% of net revenue",
                "explanation": "Fixed shipping/marketing costs on low-value orders and deep discounts can push individual orders below zero.",
                "action": "Set a minimum order value for free shipping and a margin floor on discounts.",
                "impact_inr": -float(low_val.C.iloc[0]),
                "impact_basis": "Losses on these transactions (recovered if they broke even).",
                "kpi": "Count of negative-contribution transactions", "period": "Next 3 months",
                "evidence_n": int(low_val.N.iloc[0])})

        # 7. Unusual monthly changes (|z| of MoM change > 2)
        mom = mon.NET_REVENUE_INR.pct_change()
        z = (mom - mom.mean()) / mom.std(ddof=1) if mom.std(ddof=1) else mom * 0
        for i in mon.index[z.abs() > 2]:
            issues.append({
                "category": "anomaly", "issue": "Unusual monthly revenue change", "dimension": "Month",
                "affected": str(pd.to_datetime(mon.SALE_MONTH[i]).date())[:7],
                "evidence": f"Net revenue {_inr(mon.NET_REVENUE_INR[i])}, {mom[i]:+.1%} vs prior month (z = {z[i]:.1f})",
                "explanation": "May reflect seasonality (e.g. year-end demand) or one-off large orders - only one year of data, so seasonality cannot be confirmed.",
                "action": "Investigate drivers (product, region, promotion) of the month; plan inventory if seasonal.",
                "impact_inr": None, "impact_basis": "Not calculable - diagnostic finding.",
                "kpi": "Monthly net revenue", "period": "Same month next year", "evidence_n": int(mon.TXN_COUNT[i])})

    for it in issues:
        it["effort"] = EFFORT.get(it["category"], "Medium")
        it["uncertainty"] = _uncertainty(it["evidence_n"])
    return rank(issues)


def rank(issues: list[dict]) -> list[dict]:
    """Score = impact (normalised) x evidence weight / effort weight."""
    imp = [i["impact_inr"] or 0 for i in issues]
    mx = max(imp) if imp and max(imp) > 0 else 1
    ew = {"Low": 1.0, "Medium": 1.5, "High": 2.0}
    uw = {"Low": 1.0, "Medium": 0.75, "High": 0.5}
    for i in issues:
        i["score"] = round(100 * max(i["impact_inr"] or 0, 0) / mx * uw[i["uncertainty"]] / ew[i["effort"]], 1)
    issues.sort(key=lambda i: i["score"], reverse=True)
    for n, i in enumerate(issues, 1):
        i["rank"] = n
    return issues


def to_frame(issues: list[dict]) -> pd.DataFrame:
    cols = ["rank", "issue", "dimension", "affected", "evidence", "explanation", "action", "impact_inr",
            "impact_basis", "effort", "uncertainty", "score", "kpi", "period", "evidence_n"]
    return pd.DataFrame(issues, columns=cols)
