"""Driver-based scenario engine (pure Python, unit tested).

Contribution is always rebuilt from revenue and cost drivers:

  gross        = units x list_price x (1 + price_change)
  discount     = gross x discount_rate
  revenue      = gross - discount
  returns      = revenue x return_rate          (return_rate = returned value / revenue)
  net_revenue  = revenue - returns
  cogs         = units x cogs_per_unit x (1 + cogs_change)
  shipping     = txns x shipping_per_txn x (1 + shipping_change)
  marketing    = txns x marketing_per_txn x (1 + marketing_change)
  contribution = net_revenue - cogs - shipping - marketing

As in the source data, returned units still carry COGS/shipping/marketing
(no inventory recovery is credited), so the model is conservative on returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


def _div(a, b):
    return a / b if b else 0.0


@dataclass
class Drivers:
    units: float
    txns: float
    list_price: float          # gross sales / units
    discount_rate: float       # discount / gross
    return_rate: float         # returned value / revenue
    cogs_per_unit: float
    shipping_per_txn: float
    marketing_per_txn: float

    @classmethod
    def from_totals(cls, t: dict) -> "Drivers":
        g = float(t.get("GROSS_SALES_INR") or 0)
        u = float(t.get("UNITS") or 0)
        n = float(t.get("TXN_COUNT") or 0)
        rev = float(t.get("REVENUE_INR") or (g - float(t.get("DISCOUNT_INR") or 0)))
        return cls(
            units=u, txns=n, list_price=_div(g, u),
            discount_rate=_div(float(t.get("DISCOUNT_INR") or 0), g),
            return_rate=_div(float(t.get("RETURNS_INR") or 0), rev),
            cogs_per_unit=_div(float(t.get("COGS_INR") or 0), u),
            shipping_per_txn=_div(float(t.get("SHIPPING_INR") or 0), n),
            marketing_per_txn=_div(float(t.get("MARKETING_INR") or 0), n),
        )


@dataclass
class Adjustments:
    volume_pct: float = 0.0          # % change in units and transactions
    price_pct: float = 0.0           # % change in list price
    discount_pp: float = 0.0         # percentage-point change in discount rate
    return_pp: float = 0.0           # percentage-point change in return rate
    cogs_pct: float = 0.0            # % change in unit cost
    shipping_pct: float = 0.0        # % change in shipping cost per txn
    marketing_pct: float = 0.0       # % change in marketing cost per txn


SCENARIO_PRESETS = {
    "Conservative": Adjustments(volume_pct=-10, price_pct=0, discount_pp=1, return_pp=1, cogs_pct=3, shipping_pct=5, marketing_pct=0),
    "Base case": Adjustments(),
    "Optimistic": Adjustments(volume_pct=8, price_pct=2, discount_pp=-1, return_pp=-1, cogs_pct=-1, shipping_pct=0, marketing_pct=5),
}


def compute(d: Drivers, a: Adjustments | None = None) -> dict:
    a = a or Adjustments()
    warnings = []
    units = d.units * (1 + a.volume_pct / 100)
    txns = d.txns * (1 + a.volume_pct / 100)
    price = d.list_price * (1 + a.price_pct / 100)
    disc_rate = d.discount_rate + a.discount_pp / 100
    ret_rate = d.return_rate + a.return_pp / 100
    if not 0 <= disc_rate <= 1:
        warnings.append(f"Discount rate {disc_rate:.1%} clipped to 0-100%")
        disc_rate = min(max(disc_rate, 0), 1)
    if not 0 <= ret_rate <= 1:
        warnings.append(f"Return rate {ret_rate:.1%} clipped to 0-100%")
        ret_rate = min(max(ret_rate, 0), 1)
    if units < 0:
        warnings.append("Volume below zero clipped to 0")
        units = txns = 0.0
    gross = units * price
    discount = gross * disc_rate
    revenue = gross - discount
    returns = revenue * ret_rate
    net = revenue - returns
    cogs = units * d.cogs_per_unit * (1 + a.cogs_pct / 100)
    ship = txns * d.shipping_per_txn * (1 + a.shipping_pct / 100)
    mkt = txns * d.marketing_per_txn * (1 + a.marketing_pct / 100)
    contrib = net - cogs - ship - mkt
    return {
        "units": units, "txns": txns, "gross_sales": gross, "discount": discount, "revenue": revenue,
        "returns": returns, "net_revenue": net, "cogs": cogs, "shipping": ship, "marketing": mkt,
        "contribution": contrib, "margin_pct": _div(contrib, net) * 100 if net else None,
        "warnings": warnings,
    }


def compare(d: Drivers, a: Adjustments) -> pd.DataFrame:
    base, prop = compute(d), compute(d, a)
    rows = []
    for k, label in [("net_revenue", "Net revenue"), ("gross_sales", "Gross sales"), ("discount", "Discounts"),
                     ("returns", "Returns"), ("cogs", "COGS"), ("shipping", "Shipping"),
                     ("marketing", "Marketing"), ("contribution", "Estimated contribution"),
                     ("margin_pct", "Contribution margin %")]:
        b, p = base[k], prop[k]
        rows.append({"Metric": label, "Baseline": b, "Scenario": p,
                     "Difference": (p - b) if (b is not None and p is not None) else None})
    return pd.DataFrame(rows)


def break_even(d: Drivers, a: Adjustments | None = None) -> dict:
    """Break-even conditions (contribution = 0), holding the other assumptions fixed."""
    a = a or Adjustments()
    out = {}
    # 1) max discount rate: net(disc) = cost  ->  gross*(1-x)*(1-r) = cost
    s = compute(d, a)
    cost = s["cogs"] + s["shipping"] + s["marketing"]
    ret_rate = min(max(d.return_rate + a.return_pp / 100, 0), 0.9999)
    if s["gross_sales"] > 0:
        x = 1 - cost / (s["gross_sales"] * (1 - ret_rate))
        out["max_discount_rate_pct"] = x * 100
    # 2) max return rate
    if s["revenue"] > 0:
        out["max_return_rate_pct"] = (1 - cost / s["revenue"]) * 100
    # 3) min list-price change: contribution is linear in price
    unit_net_per_price = s["units"] * (1 - (s["discount"] / s["gross_sales"] if s["gross_sales"] else 0)) * (1 - ret_rate)
    if unit_net_per_price > 0:
        p_be = cost / unit_net_per_price
        out["break_even_list_price_inr"] = p_be
        out["price_change_to_break_even_pct"] = (p_be / d.list_price - 1) * 100 if d.list_price else None
    # 4) max COGS increase
    base_cogs = d.units * d.cogs_per_unit * (1 + a.volume_pct / 100)
    if base_cogs > 0:
        out["max_cogs_increase_pct"] = ((s["net_revenue"] - s["shipping"] - s["marketing"]) / base_cogs - 1) * 100
    return out


def to_dict(a: Adjustments) -> dict:
    return asdict(a)
