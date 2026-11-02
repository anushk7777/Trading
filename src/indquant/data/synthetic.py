"""Synthetic NSE-like market generator.

Used for tests and for demonstrating the pipeline without network access. The generator
can *plant* a persistent per-stock drift (which creates momentum-type predictability) or
produce a pure random walk (``alpha_strength=0``) where no strategy should have an edge.
That second mode is an important guard: if a backtest finds "alpha" in pure noise, the
pipeline is leaking future information or overfitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MarketData

SECTORS = [
    "Financial Services", "Information Technology", "Oil Gas & Consumable Fuels",
    "Fast Moving Consumer Goods", "Automobile and Auto Components", "Healthcare",
    "Capital Goods", "Metals & Mining", "Chemicals", "Consumer Durables",
]


def generate_synthetic_market(
    n_stocks: int = 150,
    n_days: int = 2520,
    start: str = "2015-01-01",
    seed: int = 42,
    alpha_strength: float = 1.0,
    alpha_annual_std: float = 0.10,
    alpha_halflife_days: float = 120.0,
    late_listing_frac: float = 0.10,
    delisting_frac: float = 0.05,
) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    symbols = [f"SYN{i:03d}" for i in range(n_stocks)]
    T, N = n_days, n_stocks

    # --- market with bull / bear regimes (2-state Markov chain) ---
    mu = {0: 0.14 / 252, 1: -0.25 / 252}
    vol = {0: 0.13 / np.sqrt(252), 1: 0.30 / np.sqrt(252)}
    p_switch = {0: 1 / 500, 1: 1 / 100}
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        prev = regime[t - 1]
        regime[t] = 1 - prev if rng.random() < p_switch[prev] else prev
    mkt = np.array([rng.normal(mu[r], vol[r]) for r in regime])

    # --- sectors ---
    sector_idx = rng.integers(0, len(SECTORS), size=N)
    sector_ret = rng.normal(0, 0.08 / np.sqrt(252), size=(T, len(SECTORS)))

    # --- stock parameters ---
    beta = rng.uniform(0.6, 1.4, size=N)
    idio_vol = rng.uniform(0.18, 0.45, size=N) / np.sqrt(252)

    # --- persistent latent drift (source of planted predictability) ---
    phi = np.exp(-np.log(2) / alpha_halflife_days)
    a_std = alpha_strength * alpha_annual_std / 252
    alpha = np.zeros((T, N))
    if a_std > 0:
        innov_std = a_std * np.sqrt(1 - phi ** 2)
        alpha[0] = rng.normal(0, a_std, size=N)
        shocks = rng.normal(0, innov_std, size=(T, N))
        for t in range(1, T):
            alpha[t] = phi * alpha[t - 1] + shocks[t]

    eps = rng.normal(0, 1, size=(T, N)) * idio_vol
    rets = beta * mkt[:, None] + sector_ret[:, sector_idx] + alpha + eps
    rets -= 0.5 * (idio_vol ** 2)  # drift correction so log-returns are not biased upward

    # --- listing / delisting (survivorship realism) ---
    alive = np.ones((T, N), dtype=bool)
    n_late = int(late_listing_frac * N)
    n_del = int(delisting_frac * N)
    perm = rng.permutation(N)
    for j in perm[:n_late]:
        alive[: rng.integers(T // 10, T // 2), j] = False
    for j in perm[n_late:n_late + n_del]:
        end = rng.integers(T // 2, T - 20)
        rets[max(end - 120, 0):end, j] -= 0.6 / 252  # decline before delisting
        alive[end:, j] = False

    start_px = rng.uniform(50, 2000, size=N)
    log_px = np.log(start_px) + np.cumsum(rets, axis=0)
    close = np.exp(log_px)

    # --- OHLC consistent with close ---
    prev_close = np.vstack([close[0] / np.exp(rets[0]), close[:-1]])
    day_vol = np.abs(rets) + idio_vol
    gap = 0.3 * rets + rng.normal(0, 0.2, size=(T, N)) * idio_vol
    open_ = prev_close * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.5, size=(T, N))) * day_vol)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5, size=(T, N))) * day_vol)

    adv_value = np.exp(rng.normal(np.log(30e7), 1.0, size=N))  # ~30 crore median
    volume = np.round(adv_value / close * np.exp(rng.normal(0, 0.4, size=(T, N))))

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.where(alive, arr, np.nan), index=dates, columns=symbols)

    close_df = frame(close)
    stock_ret = close_df.pct_change(fill_method=None)
    ew = stock_ret.mean(axis=1).fillna(0.0)
    index = pd.Series(1000.0 * np.cumprod(1 + ew.to_numpy()), index=dates, name="INDEX")

    return MarketData(
        open=frame(open_), high=frame(high), low=frame(low), close=close_df, volume=frame(volume),
        benchmark=index.copy(), index=index,
        sectors=pd.Series([SECTORS[i] for i in sector_idx], index=symbols, dtype=object),
        membership=None, source="synthetic", survivorship_free=True,
    )
