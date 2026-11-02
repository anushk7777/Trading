"""Risk management: covariance estimation, volatility targeting, regime and drawdown control."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrinkage covariance (Ledoit-Wolf) - far more stable than the sample covariance
    when the number of stocks is large relative to the history length."""
    from sklearn.covariance import LedoitWolf

    r = returns.dropna(axis=1, thresh=max(20, int(0.8 * len(returns)))).fillna(0.0)
    if r.shape[1] == 0 or r.shape[0] < 20:
        return pd.DataFrame()
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def portfolio_vol(weights: pd.Series, returns: pd.DataFrame) -> float:
    """Annualised ex-ante volatility of ``weights`` using a Ledoit-Wolf covariance."""
    if weights.empty:
        return 0.0
    cov = ledoit_wolf_cov(returns[weights.index.intersection(returns.columns)])
    if cov.empty:
        return float("nan")
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    var = float(w @ cov.to_numpy() @ w)
    return float(np.sqrt(max(var, 0.0) * 252))


def vol_target_multiplier(predicted_vol: float, target_vol: float, max_leverage: float = 1.0) -> float:
    if not predicted_vol or np.isnan(predicted_vol) or predicted_vol <= 0:
        return 1.0
    return float(min(max_leverage, target_vol / predicted_vol))


def regime_is_risk_on(index: pd.Series, i: int, ma: int) -> bool:
    if i < ma:
        return True
    window = index.iloc[i - ma + 1: i + 1]
    return bool(index.iloc[i] >= window.mean())


def drawdown_multiplier(equity: float, peak: float, threshold: float, multiplier: float) -> float:
    if not peak or peak <= 0 or threshold is None:
        return 1.0
    dd = 1 - equity / peak
    return float(multiplier) if dd > threshold else 1.0
