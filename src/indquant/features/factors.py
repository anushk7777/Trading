"""Equity factors (higher value = more attractive). All computed causally.

Price-based factors follow the academic / practitioner literature used by systematic
managers: Jegadeesh-Titman momentum (12-1), residual (idiosyncratic) momentum
(Blitz-Huij-Martens), low volatility / betting-against-beta (Frazzini-Pedersen),
52-week-high proximity (George-Hwang), short-term reversal and trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import cs_rank


def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    y = pd.DataFrame(np.repeat(mkt.to_numpy()[:, None], ret.shape[1], axis=1), index=ret.index, columns=ret.columns)
    y = y.where(ret.notna())
    x = ret.where(y.notna())
    mx = x.rolling(window, min_periods=min_periods).mean()
    my = y.rolling(window, min_periods=min_periods).mean()
    cov = (x * y).rolling(window, min_periods=min_periods).mean() - mx * my
    var = (y * y).rolling(window, min_periods=min_periods).mean() - my * my
    return cov / var.replace(0, np.nan)


def compute_factors(data) -> dict[str, pd.DataFrame]:
    c = data.close
    ret = c.pct_change(fill_method=None)
    mret = data.index.pct_change(fill_method=None)
    beta = rolling_beta(ret, mret)

    resid = ret - beta.mul(mret, axis=0)
    idio_mom = resid.rolling(231, min_periods=200).sum().shift(21) / resid.rolling(252, min_periods=200).std()

    factors = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "mom_6_1": c.shift(21) / c.shift(126) - 1,
        "idio_mom": idio_mom,
        "low_vol": -ret.rolling(252, min_periods=200).std(),
        "low_beta": -beta,
        "reversal_1m": -(c / c.shift(21) - 1),
        "trend": c / c.rolling(200, min_periods=200).mean() - 1,
        "high_52w": c / c.rolling(252, min_periods=200).max() - 1,
    }
    from .own_signals import own_factors

    factors.update(own_factors(data))
    if getattr(data, "alt_data", None):
        from .smart_money import smart_money_factors

        factors.update(smart_money_factors(data))
    f = data.fundamentals or {}
    if "roe" in f and "debt_to_equity" in f:
        factors["quality"] = cs_rank(f["roe"]) - cs_rank(f["debt_to_equity"])
    elif "roe" in f:
        factors["quality"] = f["roe"]
    value_parts = [cs_rank(f[k]) for k in ("earnings_yield", "book_to_price") if k in f]
    if value_parts:
        factors["value"] = sum(value_parts) / len(value_parts)
    return {k: v.where(c.notna()) for k, v in factors.items()}
