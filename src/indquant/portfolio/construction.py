"""Portfolio construction: selection with turnover buffer, weighting and constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_with_buffer(scores: pd.Series, current: list[str], top_n: int, buffer_mult: float) -> list[str]:
    """Pick the top ``top_n`` names, but keep existing holdings while they rank within
    ``top_n * buffer_mult``. This hysteresis is the main turnover-control tool used by
    systematic managers - it avoids paying costs for tiny rank changes."""
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    limit = top_n * buffer_mult
    keep = sorted((s for s in current if s in rank.index and rank[s] <= limit), key=lambda s: rank[s])[:top_n]
    selection = list(keep)
    for sym in ranked.index:
        if len(selection) >= top_n:
            break
        if sym not in selection:
            selection.append(sym)
    return sorted(selection, key=lambda s: rank[s])


def raw_weights(selection: list[str], vol: pd.Series | None, method: str) -> pd.Series:
    if not selection:
        return pd.Series(dtype=float)
    if method == "inverse_vol" and vol is not None:
        v = vol.reindex(selection)
        v = v.fillna(v.median() if v.notna().any() else 1.0).clip(lower=1e-4)
        w = 1.0 / v
    else:
        w = pd.Series(1.0, index=selection)
    return w / w.sum()


def apply_caps(weights: pd.Series, max_weight: float, sector_cap: float | None = None,
               sectors: pd.Series | None = None, iterations: int = 200) -> pd.Series:
    """Enforce per-name and per-sector caps by iterative water-filling: capped names and
    sectors are frozen and the excess is redistributed pro-rata to the remaining names.
    If the constraints are infeasible the remainder stays in cash (sum < 1)."""
    if weights.empty:
        return weights.astype(float)
    raw = weights.astype(float) / weights.sum()
    w = raw.copy()
    sec = sectors.reindex(w.index).fillna("Unknown") if sectors is not None else None
    frozen = pd.Series(False, index=w.index)
    for _ in range(iterations):
        over = w > max_weight
        w[over] = max_weight
        frozen |= w >= max_weight - 1e-12
        if sec is not None and sector_cap:
            sec_sum = w.groupby(sec).sum()
            for name, total in sec_sum.items():
                members = sec.index[sec == name]
                if total > sector_cap + 1e-12:
                    w[members] *= sector_cap / total
                if w[members].sum() >= sector_cap - 1e-12:
                    frozen[members] = True
        deficit = 1.0 - w.sum()
        free = ~frozen
        if deficit <= 1e-10 or not free.any():
            break
        w[free] += deficit * raw[free] / raw[free].sum()
    return w


def target_shares(weights: pd.Series, equity: float, prices: pd.Series) -> pd.Series:
    px = prices.reindex(weights.index)
    shares = np.floor((weights * equity) / px).fillna(0)
    return shares.astype(int)
