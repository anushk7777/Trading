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
