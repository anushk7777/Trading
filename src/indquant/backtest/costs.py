"""Indian equity delivery transaction cost and slippage model.

Statutory charges (STT, stamp duty, exchange, SEBI, GST) are what NSE delivery trades
actually pay. Rates change from time to time, so all of them are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    dp: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst + self.dp


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.0
    brokerage_max_per_order: float = 20.0
    stt_buy_pct: float = 0.001
    stt_sell_pct: float = 0.001
    exchange_txn_pct: float = 0.0000297
    sebi_fee_pct: float = 0.000001
    stamp_duty_buy_pct: float = 0.00015
    gst_pct: float = 0.18
    dp_charge_per_sell: float = 15.93
    slippage_bps: float = 5.0
    impact_coef: float = 0.1
    max_slippage: float = 0.02

    @classmethod
    def from_config(cls, costs_cfg: dict) -> "CostModel":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: float(v) for k, v in costs_cfg.items() if k in fields})

    def breakdown(self, side: str, value: float) -> CostBreakdown:
        """Charges for one order of traded ``value`` (INR). ``side`` is 'buy' or 'sell'."""
        if value <= 0:
            return CostBreakdown(0, 0, 0, 0, 0, 0, 0)
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side}")
        brokerage = min(self.brokerage_pct * value, self.brokerage_max_per_order) if self.brokerage_pct > 0 else 0.0
        stt = value * (self.stt_buy_pct if side == "buy" else self.stt_sell_pct)
        exchange = value * self.exchange_txn_pct
        sebi = value * self.sebi_fee_pct
        stamp = value * self.stamp_duty_buy_pct if side == "buy" else 0.0
        gst = self.gst_pct * (brokerage + exchange + sebi)
        dp = self.dp_charge_per_sell if side == "sell" else 0.0
        return CostBreakdown(brokerage, stt, exchange, sebi, stamp, gst, dp)

    def cost(self, side: str, value: float) -> float:
        return self.breakdown(side, value).total

    def proportional_rate(self, side: str) -> float:
        """Approximate proportional cost rate (ignores fixed DP charge / brokerage cap)."""
        rate = (self.stt_buy_pct if side == "buy" else self.stt_sell_pct) + self.exchange_txn_pct + self.sebi_fee_pct
        rate += self.gst_pct * (self.exchange_txn_pct + self.sebi_fee_pct + self.brokerage_pct)
        rate += self.brokerage_pct
        if side == "buy":
            rate += self.stamp_duty_buy_pct
        return rate

    def slippage(self, order_value: float, adv_value: float, daily_vol: float) -> float:
        """Fractional price slippage: fixed half-spread plus square-root market impact."""
        slip = self.slippage_bps / 1e4
        if adv_value and adv_value > 0 and daily_vol and not math.isnan(daily_vol) and not math.isnan(adv_value):
            slip += self.impact_coef * daily_vol * math.sqrt(max(order_value, 0.0) / adv_value)
        return min(slip, self.max_slippage)
