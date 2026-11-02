"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, López de Prado & Zhu (2015). Given the daily returns of N tried
configurations, split time into S blocks; for every way of choosing half the blocks as
"in-sample", pick the configuration with the best in-sample Sharpe and see where it
ranks out-of-sample. PBO is the share of splits in which the in-sample winner lands
below the out-of-sample median. PBO near 0 = the selection process finds something
real; PBO >= 0.5 = picking the best backtest is no better than picking at random.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _sharpe(block: np.ndarray) -> np.ndarray:
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, block.mean(axis=0) / sd, 0.0)


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_blocks: int = 10) -> dict:
    """``returns``: T x N daily returns, one column per configuration tried."""
    r = returns.dropna(how="any").to_numpy(dtype=float)
    T, N = r.shape
    if N < 2:
        raise ValueError("PBO needs at least two configurations")
    n_blocks = max(2, n_blocks - n_blocks % 2)
    if T < n_blocks * 5:
        raise ValueError("not enough observations for PBO")
    blocks = np.array_split(np.arange(T), n_blocks)
    logits, oos_of_best = [], []
    for ins in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[k] for k in ins])
        oos_idx = np.concatenate([blocks[k] for k in range(n_blocks) if k not in ins])
        sr_is, sr_oos = _sharpe(r[is_idx]), _sharpe(r[oos_idx])
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        oos_of_best.append(sr_oos[best] * np.sqrt(252))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "n_configs": N,
            "median_logit": float(np.median(logits)), "oos_sharpe_of_is_best": float(np.median(oos_of_best))}
