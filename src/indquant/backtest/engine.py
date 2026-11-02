"""Daily event-driven backtest engine.

Timeline for every bar ``i``:
  1. pending orders (created at the close of bar i-1) fill at the OPEN of bar i,
     sells before buys, with slippage + Indian statutory charges, integer shares,
     and buys limited by available cash;
  2. bracket (stop / target) exits are checked against bar i's open/high/low - a gap
     through the level fills at the open, and if both levels are touched inside the
     bar the stop is assumed to have been hit first (conservative);
  3. positions without prices for many sessions (delisted / suspended) are closed;
  4. idle cash accrues the configured yield; equity is marked at the close;
  5. the strategy sees data up to the close of bar i and returns new orders.

Because orders always fill on the *next* bar, a strategy can never trade on the same
price it used to make its decision.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd

from .account import Account, Order, apply_buy, apply_sell
from .costs import CostModel

STALE_DAYS = 10


class Strategy(Protocol):
    name: str

    def prepare(self, data) -> None: ...

    def on_bar(self, i: int, ctx: "BarContext") -> list[Order]: ...


class BarContext:
    def __init__(self, engine: "Engine", i: int, account: Account, equity: float):
        self.engine = engine
        self.i = i
        self.date = engine.dates[i]
        self.account = account
        self.equity = equity

    @property
    def close(self) -> pd.Series:
        return self.engine.close_ff.iloc[self.i]


class Engine:
    def __init__(self, data, cost_model: CostModel, cash_yield: float = 0.0):
        self.data = data
        self.costs = cost_model
        self.dates = data.dates
        self.symbols = data.symbols
        self.col = {s: j for j, s in enumerate(self.symbols)}
        self.O = data.open.to_numpy(dtype=float)
        self.H = data.high.to_numpy(dtype=float)
        self.L = data.low.to_numpy(dtype=float)
        self.C = data.close.to_numpy(dtype=float)
        self.V = data.volume.to_numpy(dtype=float)
        self.close_ff = data.close.ffill()
        self.adv = (data.close * data.volume).rolling(21, min_periods=5).mean().to_numpy(dtype=float)
        self.dvol = data.close.pct_change(fill_method=None).rolling(21, min_periods=10).std().to_numpy(dtype=float)
        self.daily_yield = (1 + cash_yield) ** (1 / 252) - 1

    # ------------------------------------------------------------------ helpers
    def _slip(self, i: int, j: int, value: float) -> float:
        k = max(i - 1, 0)  # liquidity known at the signal bar
        return self.costs.slippage(value, self.adv[k, j], self.dvol[k, j])

    def _tradable(self, i: int, j: int) -> bool:
        o, v = self.O[i, j], self.V[i, j]
        return not math.isnan(o) and o > 0 and not (not math.isnan(v) and v <= 0)

    def _sell(self, acct: Account, i: int, sym: str, shares: int, base_px: float, reason: str,
              apply_slip: bool = True) -> None:
        pos = acct.positions.get(sym)
        if pos is None or shares <= 0:
            return
        shares = min(shares, pos.shares)
        slip = self._slip(i, self.col[sym], shares * base_px) if apply_slip else 0.0
        px = base_px * (1 - slip)
        cost = self.costs.cost("sell", shares * px)
        apply_sell(acct, sym, shares, px, cost, self.dates[i], reason, slippage=shares * base_px * slip)

    def _buy(self, acct: Account, i: int, order: Order) -> bool:
        j = self.col.get(order.symbol)
        if j is None or not self._tradable(i, j):
            return False
        o = self.O[i, j]
        slip = self._slip(i, j, order.shares * o)
        px = o * (1 + slip)
        rate = self.costs.proportional_rate("buy")
        affordable = int(math.floor(max(acct.cash, 0.0) / (px * (1 + rate) + 1e-12)))
        shares = min(order.shares, affordable)
        if shares <= 0:
            return True  # nothing affordable - drop the order
        cost = self.costs.cost("buy", shares * px)
        if shares * px + cost > acct.cash:  # brokerage minimums can tip it over
            shares -= 1
            if shares <= 0:
                return True
            cost = self.costs.cost("buy", shares * px)
        apply_buy(acct, order.symbol, shares, px, cost, self.dates[i], order.stop_dist, order.target_dist,
                  order.atr, order.reason, slippage=shares * o * slip)
        return True

    # --------------------------------------------------------------- main step
    def step(self, i: int, strategy, acct: Account) -> None:
        pending, acct.pending = acct.pending, []
        carry: list[Order] = []
        sells = [o for o in pending if o.shares < 0]
        buys = [o for o in pending if o.shares > 0]
        for o in sells:
            j = self.col.get(o.symbol)
            if o.symbol not in acct.positions:
                continue
            if j is None or not self._tradable(i, j):
                if o.ttl > 1:
                    o.ttl -= 1
                    carry.append(o)
                continue
            self._sell(acct, i, o.symbol, -o.shares, self.O[i, j], o.reason)
        for o in buys:
            if not self._buy(acct, i, o) and o.ttl > 1:
                o.ttl -= 1
                carry.append(o)

        # bracket exits
        for sym in list(acct.positions):
            pos = acct.positions[sym]
            if pos.stop is None and pos.target is None:
                continue
            j = self.col[sym]
            o, h, l = self.O[i, j], self.H[i, j], self.L[i, j]
            if math.isnan(o) or math.isnan(l) or math.isnan(h):
                continue
            if pos.stop is not None and o <= pos.stop:
                self._sell(acct, i, sym, pos.shares, o, "stop_gap")
            elif pos.target is not None and o >= pos.target:
                self._sell(acct, i, sym, pos.shares, o, "target_gap")
            elif pos.stop is not None and l <= pos.stop:
                self._sell(acct, i, sym, pos.shares, pos.stop, "stop")
            elif pos.target is not None and h >= pos.target:
                self._sell(acct, i, sym, pos.shares, pos.target, "target", apply_slip=False)

        # stale / delisted positions and last prices
        for sym in list(acct.positions):
            pos = acct.positions[sym]
            c = self.C[i, self.col[sym]]
            if math.isnan(c):
                pos.missing_days += 1
                if pos.missing_days >= STALE_DAYS:
                    px = pos.last_price if pos.last_price else pos.entry_price
                    self._sell(acct, i, sym, pos.shares, px, "delisted")
            else:
                pos.missing_days = 0
                pos.last_price = c

        if acct.cash > 0:
            acct.cash *= 1 + self.daily_yield
        close = self.close_ff.iloc[i]
        equity = acct.equity(close)
        acct.peak_equity = max(acct.peak_equity, equity)
        invested = equity - acct.cash
        acct.history.append({"date": self.dates[i], "equity": equity, "cash": acct.cash,
                             "exposure": invested / equity if equity > 0 else 0.0,
                             "n_positions": len(acct.positions)})
        acct.last_date = str(self.dates[i].date())

        new_orders = strategy.on_bar(i, BarContext(self, i, acct, equity)) if strategy is not None else []
        new_syms = {o.symbol for o in new_orders}
        acct.pending = [o for o in carry if o.symbol not in new_syms] + list(new_orders)

    def run(self, strategy, acct: Account, start_idx: int, end_idx: int | None = None) -> Account:
        end_idx = len(self.dates) if end_idx is None else end_idx
        for i in range(start_idx, end_idx):
            self.step(i, strategy, acct)
        return acct


def history_frame(acct: Account) -> pd.DataFrame:
    if not acct.history:
        return pd.DataFrame(columns=["equity", "cash", "exposure", "n_positions"])
    return pd.DataFrame(acct.history).set_index("date")


def trades_frame(accounts: list[Account]) -> pd.DataFrame:
    rows = [t for a in accounts for t in a.trades]
    if not rows:
        return pd.DataFrame(columns=["date", "sleeve", "symbol", "side", "shares", "price", "value", "costs",
                                     "slippage", "reason", "pnl", "holding_days", "tax_lots"])
    return pd.DataFrame(rows)


def equity_values(df: pd.DataFrame) -> np.ndarray:
    return df["equity"].to_numpy(dtype=float)
