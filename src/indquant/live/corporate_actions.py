"""Corporate actions for live / paper books held across days.

Adjusted price sources (Yahoo auto-adjust, the NSE bhavcopy pipeline) rewrite *history*
when a split, bonus or dividend goes ex: yesterday's close is rescaled, today's price is
real. A book that stored "13 shares, last price 2,990" must therefore be reconciled, or a
1:1 bonus looks like a 50% loss.

Each position remembers the date and price of its last mark. On the next run, the newly
loaded close for that same date is compared with the stored one:
* ratio far from 1 (< 0.85 or > 1.15) -> split / bonus / consolidation: shares, lot costs,
  stop and target are rescaled by the (rationalised) ratio;
* small ratio below 1 -> dividend back-adjustment: the dividend is credited as cash;
* tiny differences are ignored.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd


def reconcile(acct, data, tol: float = 0.001) -> list[dict]:
    events = []
    close = data.close
    for sym, pos in list(acct.positions.items()):
        if pos.last_price is None or not pos.last_date or sym not in close.columns:
            continue
        d = pd.Timestamp(pos.last_date)
        k = int(close.index.searchsorted(d, side="right")) - 1
        if k < 0 or close.index[k] != d:
            continue
        now = close[sym].iloc[k]
        if np.isnan(now) or not pos.last_price:
            continue
        ratio = float(now / pos.last_price)
        if abs(ratio - 1) < tol:
            continue
        if ratio < 0.85 or ratio > 1.15:
            mult = Fraction(1 / ratio).limit_denominator(20)       # e.g. 2 for a 1:1 bonus, 5 for a 1:5 split
            m = float(mult)
            old = pos.shares
            for lot in pos.lots:
                lot.shares = int(round(lot.shares * m))
                lot.cost /= m
            pos.shares = sum(lot.shares for lot in pos.lots)
            for attr in ("entry_price", "stop", "target", "atr"):
                if getattr(pos, attr) is not None:
                    setattr(pos, attr, getattr(pos, attr) / m)
            pos.last_price = float(now)
            events.append({"symbol": sym, "kind": "split", "multiplier": str(mult), "shares_before": old,
                           "shares_after": pos.shares,
                           "message": f"{sym}: split/bonus detected (x{mult}) - shares {old} -> {pos.shares}; "
                                      "cost, stop and target rescaled. Update the GTT at your broker."})
        elif ratio < 1:
            per_share = pos.last_price * (1 - ratio)
            amount = per_share * pos.shares
            acct.cash += amount
            pos.last_price = float(now)
            events.append({"symbol": sym, "kind": "dividend", "per_share": round(per_share, 4),
                           "amount": round(amount, 2),
                           "message": f"{sym}: dividend of about ₹{per_share:,.2f}/share credited (₹{amount:,.0f})."})
        else:
            pos.last_price = float(now)
    return events
