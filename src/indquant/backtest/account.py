"""Account ledger: cash, integer-share positions with FIFO tax lots, pending orders."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Lot:
    date: str
    shares: int
    cost: float  # per share, including buy-side charges


@dataclass
class Position:
    symbol: str
    shares: int
    lots: list[Lot]
    entry_date: str
    entry_price: float
    stop: float | None = None
    target: float | None = None
    atr: float | None = None
    missing_days: int = 0
    last_price: float | None = None
    realized: float = 0.0  # realised P&L from partial exits of this position
    last_date: str | None = None  # date of last_price (needed to reconcile corporate actions)

    @property
    def avg_cost(self) -> float:
        total = sum(l.shares for l in self.lots)
        return sum(l.shares * l.cost for l in self.lots) / total if total else 0.0


@dataclass
class Order:
    symbol: str
    shares: int                      # >0 buy, <0 sell
    reason: str
    created: str
    ttl: int = 1                     # sessions the order stays valid if it cannot fill
    stop_dist: float | None = None   # bracket distances (set at fill relative to fill price)
    target_dist: float | None = None
    atr: float | None = None
    ref_price: float | None = None   # signal-day close, for reporting

    @property
    def side(self) -> str:
        return "buy" if self.shares > 0 else "sell"


@dataclass
class Account:
    name: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    peak_equity: float = 0.0
    last_date: str | None = None

    def market_value(self, prices: pd.Series) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym, float("nan"))
            if px != px:  # NaN -> last known price
                px = pos.last_price if pos.last_price is not None else pos.entry_price
            total += pos.shares * px
        return total

    def equity(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)

    # --- persistence -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name, "cash": self.cash, "peak_equity": self.peak_equity, "last_date": self.last_date,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "pending": [asdict(o) for o in self.pending],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        positions = {}
        for k, v in d.get("positions", {}).items():
            v = dict(v)
            v["lots"] = [Lot(**l) for l in v["lots"]]
            positions[k] = Position(**v)
        return cls(
            name=d["name"], cash=float(d["cash"]), positions=positions,
            pending=[Order(**o) for o in d.get("pending", [])], meta=d.get("meta", {}),
            peak_equity=float(d.get("peak_equity", 0.0)), last_date=d.get("last_date"),
        )


# ---------------------------------------------------------------------------------
# Fill bookkeeping shared by the backtest engine, the paper book and the live desk,
# so a real fill is booked exactly the way a simulated one is.
# ---------------------------------------------------------------------------------
def apply_buy(acct: "Account", symbol: str, shares: int, price: float, cost: float, date,
              stop_dist: float | None = None, target_dist: float | None = None, atr: float | None = None,
              reason: str = "", slippage: float = 0.0) -> dict:
    value = shares * price
    acct.cash -= value + cost
    day = str(pd.Timestamp(date).date())
    pos = acct.positions.get(symbol)
    if pos is None:
        pos = Position(symbol=symbol, shares=0, lots=[], entry_date=day, entry_price=price, atr=atr)
        if stop_dist:
            pos.stop = price - stop_dist
        if target_dist:
            pos.target = price + target_dist
        acct.positions[symbol] = pos
    pos.shares += shares
    pos.lots.append(Lot(date=day, shares=shares, cost=(value + cost) / shares))
    if pos.last_date is None:
        # a fill price is not a closing-price mark: leave last_date empty so corporate-action
        # reconciliation only ever compares closes with closes
        pos.last_price = price
    trade = {"date": pd.Timestamp(date), "sleeve": acct.name, "symbol": symbol, "side": "buy", "shares": shares,
             "price": price, "value": value, "costs": cost, "slippage": slippage, "reason": reason,
             "pnl": None, "holding_days": None, "tax_lots": []}
    acct.trades.append(trade)
    return trade


def apply_sell(acct: "Account", symbol: str, shares: int, price: float, cost: float, date,
               reason: str = "", slippage: float = 0.0) -> dict | None:
    """Book a sale with FIFO lot matching (realised P&L, holding periods for tax)."""
    pos = acct.positions.get(symbol)
    if pos is None or shares <= 0:
        return None
    shares = min(shares, pos.shares)
    value = shares * price
    acct.cash += value - cost
    sell_date = pd.Timestamp(date)
    net_px = (value - cost) / shares
    remaining, basis, lots_used = shares, 0.0, []
    while remaining > 0 and pos.lots:
        lot = pos.lots[0]
        take = min(lot.shares, remaining)
        basis += take * lot.cost
        lots_used.append({"shares": take, "buy_date": lot.date, "gain": take * (net_px - lot.cost),
                          "days": (sell_date - pd.Timestamp(lot.date)).days})
        lot.shares -= take
        remaining -= take
        if lot.shares == 0:
            pos.lots.pop(0)
    pnl = value - cost - basis
    hold = (sell_date - pd.Timestamp(pos.entry_date)).days
    pos.shares -= shares
    trade = {"date": sell_date, "sleeve": acct.name, "symbol": symbol, "side": "sell", "shares": shares,
             "price": price, "value": value, "costs": cost, "slippage": slippage, "reason": reason,
             "pnl": pnl, "holding_days": hold, "tax_lots": lots_used}
    acct.trades.append(trade)
    if pos.shares <= 0:
        del acct.positions[symbol]
        # bookkeeping used by the risk governor and re-entry filters
        total_pnl = pos.realized + pnl
        acct.meta["loss_streak"] = int(acct.meta.get("loss_streak", 0)) + 1 if total_pnl < 0 else 0
        acct.meta.setdefault("last_exit", {})[symbol] = {"date": str(sell_date.date()), "reason": reason,
                                                         "pnl": float(total_pnl)}
    else:
        pos.realized += pnl
    return trade
