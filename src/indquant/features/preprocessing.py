"""Cross-sectional preprocessing used by factor models (all operations are per-date)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-date percentile rank centred on zero, in (-0.5, 0.5). NaN outside ``mask``."""
    if mask is not None:
        df = df.where(mask)
    ranks = df.rank(axis=1, method="average")
    n = df.notna().sum(axis=1)
    return ranks.sub(0.5, axis=0).div(n.replace(0, np.nan), axis=0) - 0.5


def cs_zscore(df: pd.DataFrame, mask: pd.DataFrame | None = None, clip: float = 3.0) -> pd.DataFrame:
    if mask is not None:
        df = df.where(mask)
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def winsorize_mad(df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1) * 1.4826
    lo = med - k * mad
    hi = med + k * mad
    return df.clip(lower=lo, upper=hi, axis=0)


def sector_neutralize(df: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Subtract the per-date sector mean so scores compare stocks within their sector."""
    sec = sectors.reindex(df.columns).fillna("Unknown")
    out = df.copy()
    for name in sec.unique():
        cols = sec.index[sec == name]
        block = df[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def combine(scores: dict[str, pd.DataFrame], weights: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
    """Weighted average of score frames; NaN scores are ignored (weights renormalised)."""
    names = [n for n in scores if (isinstance(weights, pd.DataFrame) and n in weights.columns)
             or (isinstance(weights, dict) and weights.get(n, 0) != 0)]
    if not names:
        raise ValueError("No factors with non-zero weight")
    ref = scores[names[0]]
    total = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    wsum = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    for n in names:
        s = scores[n]
        if isinstance(weights, pd.DataFrame):
            w = pd.DataFrame(np.repeat(weights[n].to_numpy()[:, None], s.shape[1], axis=1),
                             index=s.index, columns=s.columns)
        else:
            w = pd.DataFrame(float(weights[n]), index=s.index, columns=s.columns)
        valid = s.notna()
        total += s.fillna(0.0) * w.where(valid, 0.0)
        wsum += w.abs().where(valid, 0.0)
    return total / wsum.replace(0, np.nan)
