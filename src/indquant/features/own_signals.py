"""indquant's own signal ideas - deliberately small, each tied to a stated hypothesis
(see research/hypotheses.py) and tested like every other factor before it gets weight.

* smooth_momentum - momentum *path quality*. Two stocks can both be up 40% in a year:
  one through steady small gains, one through a single news jump. Steady drift reflects
  information that investors absorb slowly and tends to persist (Da, Gurun & Warachka's
  "frog in the pan"); a jump is already priced. Value = sign(return) x (%up days - %down days).
* lottery_aversion - minus the largest single-day gain in the last month. Stocks with
  lottery-like spikes attract speculative buying and subsequently underperform
  (Bali, Cakici & Whitelaw's MAX effect) - very visible in Indian small/mid caps.
* circuit_days - India-specific: sessions in the last quarter that closed pinned at the
  day's high or low after a >= 4.8% move, i.e. at/near an exchange price band. Repeated
  band hits signal operator activity / thin liquidity where stops cannot fill.
  Used as a negative factor and as a swing veto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    ret = close.pct_change(fill_method=None)
    window = lookback - skip
    up = (ret > 0).astype(float).where(ret.notna()).rolling(window, min_periods=window // 2).mean().shift(skip)
    down = (ret < 0).astype(float).where(ret.notna()).rolling(window, min_periods=window // 2).mean().shift(skip)
    pret = close.shift(skip) / close.shift(lookback) - 1
    return np.sign(pret) * (up - down)


def lottery_aversion(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return -close.pct_change(fill_method=None).rolling(window, min_periods=window // 2).max()


def circuit_days(data, window: int = 63, move: float = 0.048, tol: float = 0.001) -> pd.DataFrame:
    c, h, lo = data.close, data.high, data.low
    ret = c.pct_change(fill_method=None)
    pinned_up = (ret >= move) & (c >= h * (1 - tol))
    pinned_dn = (ret <= -move) & (c <= lo * (1 + tol))
    hits = (pinned_up | pinned_dn).astype(float).where(c.notna())
    return hits.rolling(window, min_periods=1).sum()


def own_factors(data) -> dict[str, pd.DataFrame]:
    return {
        "smooth_momentum": smooth_momentum(data.close),
        "lottery_aversion": lottery_aversion(data.close),
        "circuit_risk": -circuit_days(data),
    }
