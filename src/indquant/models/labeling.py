"""Training labels. Labels look into the future by construction and must only be used
inside purged training windows (see :mod:`indquant.models.validation`)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.preprocessing import cs_rank
from ..research.factor_analysis import forward_returns


def rank_labels(data, horizon: int, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional percentile rank of the forward open-to-open return."""
    return cs_rank(forward_returns(data, horizon), mask)


@dataclass
class TripleBarrier:
    label: pd.DataFrame       # 1 = target hit first, 0 = stop hit first or time barrier, NaN = unknown
    exit_offset: pd.DataFrame # bars after signal date at which the barrier was touched
    ret: pd.DataFrame         # realised return of the barrier trade (before costs)


def _shift(arr: np.ndarray, k: int) -> np.ndarray:
    """out[t] = arr[t + k] (NaN past the end)."""
    out = np.full_like(arr, np.nan, dtype=float)
    if k < arr.shape[0]:
        out[: arr.shape[0] - k] = arr[k:]
    return out


def triple_barrier(data, atr: pd.DataFrame, stop_mult: float, target_mult: float, max_hold: int) -> TripleBarrier:
    """Triple-barrier labels (López de Prado, *Advances in Financial Machine Learning*).

    Signal at close of t, entry at open of t+1. Stop = entry - stop_mult*ATR_t and
    target = entry + target_mult*ATR_t. If both are touched within the same bar the stop
    is assumed to come first (conservative). Unresolved after ``max_hold`` bars -> exit at
    close (label 0).
    """
    O = data.open.to_numpy(dtype=float)
    H = data.high.to_numpy(dtype=float)
    L = data.low.to_numpy(dtype=float)
    C = data.close.to_numpy(dtype=float)
    A = atr.reindex_like(data.close).to_numpy(dtype=float)

    entry = _shift(O, 1)
    stop = entry - stop_mult * A
    target = entry + target_mult * A
    label = np.full(C.shape, np.nan)
    offset = np.full(C.shape, np.nan)
    rets = np.full(C.shape, np.nan)
    valid = ~np.isnan(entry) & ~np.isnan(A) & (A > 0)
    resolved = ~valid

    for k in range(1, max_hold + 1):
        hk, lk = _shift(H, k), _shift(L, k)
        avail = ~np.isnan(lk) & ~np.isnan(hk)
        stop_hit = avail & ~resolved & (lk <= stop)
        tgt_hit = avail & ~resolved & ~stop_hit & (hk >= target)
        ok = _shift(O, k)
        gap_stop_px = np.where(ok < stop, ok, stop)
        gap_tgt_px = np.where(ok > target, ok, target)
        label[stop_hit], offset[stop_hit] = 0.0, k
        rets[stop_hit] = gap_stop_px[stop_hit] / entry[stop_hit] - 1
        label[tgt_hit], offset[tgt_hit] = 1.0, k
        rets[tgt_hit] = gap_tgt_px[tgt_hit] / entry[tgt_hit] - 1
        resolved |= stop_hit | tgt_hit

    c_end = _shift(C, max_hold)
    timed = ~resolved & ~np.isnan(c_end)
    label[timed], offset[timed] = 0.0, max_hold
    rets[timed] = c_end[timed] / entry[timed] - 1

    mk = lambda a: pd.DataFrame(a, index=data.close.index, columns=data.close.columns)  # noqa: E731
    return TripleBarrier(mk(label), mk(offset), mk(rets))
