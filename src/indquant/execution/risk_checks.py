"""Hard pre-trade risk checks - the last gate before any order leaves the desk.

These run on every approval regardless of what the model, the AI reviewer or the user
says. A failing check rejects the order; it is never silently resized.
"""
from __future__ import annotations


def pre_trade_checks(p: dict, *, kill: bool, total_equity: float, held_qty: int, sleeve_cash: float,
                     orders_today: int, ex) -> list[str]:
    problems = []
    qty, price, ref = int(p["qty"]), float(p["limit_price"]), float(p["ref_price"])
    if qty <= 0:
        problems.append("quantity must be positive")
    if p["side"] == "BUY":
        if kill:
            problems.append("kill switch is ON - no new buys")
        if qty * price > float(ex.max_order_pct) * total_equity:
            problems.append(f"order value ₹{qty * price:,.0f} exceeds {float(ex.max_order_pct):.0%} of equity")
        if qty * price * 1.002 > sleeve_cash:
            problems.append(f"not enough cash in sleeve (₹{sleeve_cash:,.0f} available)")
    else:
        if qty > held_qty:
            problems.append(f"cannot sell {qty}: only {held_qty} held (no short selling)")
    if ref > 0 and abs(price / ref - 1) > float(ex.max_price_deviation):
        problems.append(f"limit ₹{price:,.2f} is more than {float(ex.max_price_deviation):.0%} from last close")
    if orders_today >= int(ex.max_orders_per_day):
        problems.append(f"daily order limit reached ({ex.max_orders_per_day})")
    return problems
