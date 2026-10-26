"""Natural-language Q&A over verified analytics -- without executing LLM SQL.

1. Cortex AI_COMPLETE converts the question to a small JSON *intent*.
2. The intent is validated against allow-lists (dimension, metric, limits).
3. A fixed, parameterised query/function from src/analytics.py runs.
4. Cortex (optional) writes a short summary that may only use the returned table.

If Cortex is unavailable, a keyword matcher produces the intent instead.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src import analytics as A
from src import forecasting as F
from src.scenarios import Adjustments, Drivers, compare

MODEL = "claude-sonnet-4-5"
METRICS = {
    "contribution": "CONTRIBUTION_INR", "net_revenue": "NET_REVENUE_INR", "margin": "MARGIN_PCT",
    "return_rate": "RETURN_RATE_PCT", "units": "UNITS", "transactions": "TXN_COUNT", "aov": "AOV_INR",
    "discount": "EFFECTIVE_DISCOUNT_PCT",
}
DIMS = {"product": "PRODUCT_NAME", "category": "CATEGORY", "region": "REGION", "channel": "SALES_CHANNEL",
        "segment": "CUSTOMER_SEGMENT", "payment": "PAYMENT_METHOD", "promotion": "PROMOTION_TYPE", "month": "SALE_MONTH"}
INTENTS = ("rank", "drivers", "investigate", "what_if", "forecast", "kpis")

PROMPT = """You convert retail analytics questions to JSON. Output ONLY JSON, no prose.
Schema: {"intent": one of ["rank","drivers","investigate","what_if","forecast","kpis"],
 "dimension": one of ["product","category","region","channel","segment","payment","promotion","month"] or null,
 "metric": one of ["contribution","net_revenue","margin","return_rate","units","transactions","aov","discount"] or null,
 "order": "desc" or "asc", "limit": integer 1-20,
 "what_if": {"discount_pp":number,"price_pct":number,"volume_pct":number,"return_pp":number,"cogs_pct":number,"shipping_pct":number,"marketing_pct":number} or null,
 "horizon": 3, 6 or 12 or null}
Rules: "rank" = top/bottom lists; "drivers" = what reduces/explains margin; "investigate" = which segment needs attention;
"what_if" = hypothetical change (decrease = negative number); "forecast" = future months; "kpis" = overall totals.
"profit" means contribution. Question: """


def _cortex(session, prompt: str) -> str | None:
    try:
        return session.sql("SELECT AI_COMPLETE(?, ?)", params=[MODEL, prompt]).collect()[0][0]
    except Exception:
        return None


def keyword_intent(q: str) -> dict:
    s = q.lower()
    dim = next((k for k in ["product", "category", "region", "channel", "segment", "payment", "promotion", "month"]
                if k in s or k + "s" in s or (k == "category" and "categories" in s)), None)
    metric = ("return_rate" if "return" in s else "margin" if "margin" in s else "units" if "best-selling" in s or "units" in s
              else "net_revenue" if "revenue" in s or "sales" in s else "contribution")
    num = re.search(r"(\d+(?:\.\d+)?)\s*(?:percentage points|pp|%|percent)", s)
    horizon = next((h for h, w in [(3, "three"), (6, "six"), (12, "twelve")] if w in s or f"{h} month" in s or f"{h}-month" in s), None)
    if "what would happen" in s or "what if" in s or "if discounts" in s:
        n = float(num.group(1)) if num else 5.0
        sign = -1 if any(w in s for w in ["decrease", "reduce", "lower", "cut"]) else 1
        key = "discount_pp" if "discount" in s else "price_pct" if "price" in s else "volume_pct" if "volume" in s else "discount_pp"
        return {"intent": "what_if", "what_if": {key: sign * n}}
    if "forecast" in s or "next" in s and "month" in s or "future" in s:
        return {"intent": "forecast", "horizon": horizon or 6}
    if "reducing" in s or "driv" in s or "hurting" in s:
        return {"intent": "drivers"}
    if "investigate" in s or "should management" in s or "attention" in s:
        return {"intent": "investigate", "dimension": dim or "region"}
    return {"intent": "rank", "dimension": dim or "product", "metric": metric,
            "order": "asc" if any(w in s for w in ["lowest", "worst", "least", "bottom"]) else "desc", "limit": 5}


def validate_intent(raw: dict) -> dict:
    """Sanitise LLM output against allow-lists (unit tested)."""
    it = raw.get("intent")
    if it not in INTENTS:
        raise ValueError(f"Unsupported intent: {it!r}")
    out = {"intent": it}
    out["dimension"] = raw.get("dimension") if raw.get("dimension") in DIMS else None
    out["metric"] = raw.get("metric") if raw.get("metric") in METRICS else "contribution"
    out["order"] = "asc" if raw.get("order") == "asc" else "desc"
    try:
        out["limit"] = max(1, min(int(raw.get("limit") or 5), 20))
    except (TypeError, ValueError):
        out["limit"] = 5
    wi = raw.get("what_if") or {}
    allowed = Adjustments.__dataclass_fields__.keys()
    alias = {"discount_pp": "discount_pp", "price_pct": "price_pct", "volume_pct": "volume_pct", "return_pp": "return_pp",
             "cogs_pct": "cogs_pct", "shipping_pct": "shipping_pct", "marketing_pct": "marketing_pct"}
    out["what_if"] = {alias[k]: max(-100.0, min(float(v), 100.0)) for k, v in wi.items()
                      if k in alias and alias[k] in allowed and isinstance(v, (int, float))}
    h = raw.get("horizon")
    out["horizon"] = h if h in F.HORIZONS else 6
    if it == "rank" and not out["dimension"]:
        out["dimension"] = "product"
    if it == "investigate" and not out["dimension"]:
        out["dimension"] = "region"
    return out


def parse_question(session, q: str) -> tuple[dict, str]:
    txt = _cortex(session, PROMPT + json.dumps(q[:500]))
    if txt:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return validate_intent(json.loads(m.group(0))), "Cortex AI_COMPLETE"
            except (ValueError, json.JSONDecodeError):
                pass
    return validate_intent(keyword_intent(q)), "keyword matcher (Cortex unavailable or unparseable)"


def answer(session, q: str, f: A.Filters | None) -> dict:
    intent, parser = parse_question(session, q)
    it = intent["intent"]
    calc, sql_note = "", ""
    if it == "rank":
        col, metric = DIMS[intent["dimension"]], METRICS[intent["metric"]]
        df = A.by_dimension(session, f, col)
        if intent["metric"] in ("return_rate", "margin"):
            df = df[df.TXN_COUNT >= 30]
            calc = "Segments with fewer than 30 transactions excluded for rate metrics."
        df = df.sort_values(metric, ascending=intent["order"] == "asc").head(intent["limit"])
        table = df[[col, metric, "NET_REVENUE_INR", "CONTRIBUTION_INR", "MARGIN_PCT", "RETURN_RATE_PCT", "TXN_COUNT"]].loc[:, lambda d: ~d.columns.duplicated()]
        sql_note = f"aggregate(GROUP BY {col}) on V_SALES_OVERVIEW, sorted by {metric} {intent['order']}"
    elif it in ("drivers", "investigate"):
        from src.recommendations import detect_issues, to_frame
        issues = to_frame(detect_issues(session, f))
        if it == "investigate":
            dim_label = {"region": "Region", "product": "Product", "category": "Category", "channel": "Sales channel",
                         "promotion": "Promotion"}.get(intent["dimension"], "Region")
            issues = issues[issues.dimension == dim_label] if (issues.dimension == dim_label).any() else issues
        table = issues[["rank", "issue", "affected", "evidence", "impact_inr", "uncertainty"]].head(8)
        sql_note = "Rule-based findings (src/recommendations.py) over filtered aggregates"
    elif it == "what_if":
        tot = A.kpis(session, f)
        adj = Adjustments(**intent["what_if"])
        table = compare(Drivers.from_totals(tot), adj)
        calc = f"Adjustments: {intent['what_if']}. Volume held constant unless stated (no demand response modelled)."
        sql_note = "Driver model (src/scenarios.py) on filtered totals"
    elif it == "forecast":
        hist = A.monthly(session, f)
        res = F.forecast(hist, intent["horizon"], "linear_trend", session)
        table = res["table"][["SALE_MONTH", "FORECAST_NET_REVENUE_INR", "LOWER_95_INR", "UPPER_95_INR", "BASE_CASE_CONTRIBUTION_INR",
                              "CONSERVATIVE_CONTRIBUTION_INR", "OPTIMISTIC_CONTRIBUTION_INR"]]
        calc = f"Linear-trend forecast on {res['history_months']} months; contribution from trailing {res['driver_window']}-month unit economics. Estimates only."
        sql_note = "V_SALES_OVERVIEW monthly aggregate -> src/forecasting.py"
    else:
        table = pd.DataFrame([A.kpis(session, f)])
        sql_note = "aggregate() on V_SALES_OVERVIEW"

    summary = None
    if len(table):
        s = _cortex(session, "You are a retail analyst. Answer the question in 3-5 sentences using ONLY the numbers in this table "
                    "(INR). Do not invent numbers. Say 'estimate' for forecasts/what-ifs and distinguish correlation from causation.\n"
                    f"Question: {q}\nTable (CSV):\n{table.head(20).to_csv(index=False)}\nNotes: {calc}")
        summary = s.strip() if s else None
    return {"intent": intent, "parser": parser, "table": table, "summary": summary, "calc_note": calc, "source": sql_note}
