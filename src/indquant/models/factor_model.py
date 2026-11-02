"""Composite multi-factor score."""
from __future__ import annotations

import pandas as pd

from ..features.preprocessing import combine, cs_rank, sector_neutralize


def rank_factors(factors: dict[str, pd.DataFrame], mask: pd.DataFrame, sectors: pd.Series | None = None,
                 sector_neutral: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for name, f in factors.items():
        r = cs_rank(f, mask)
        if sector_neutral and sectors is not None:
            r = sector_neutralize(r, sectors)
        out[name] = r
    return out


def composite_score(ranked: dict[str, pd.DataFrame], weights: dict[str, float] | pd.DataFrame,
                    mask: pd.DataFrame) -> pd.DataFrame:
    """Weighted blend of ranked factors, re-ranked so the output is in (-0.5, 0.5)."""
    if isinstance(weights, dict):
        weights = {k: v for k, v in weights.items() if k in ranked and v}
    return cs_rank(combine(ranked, weights), mask)
