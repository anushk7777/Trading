"""Factor research: information coefficients, quantile spreads, decay (alphalens-style)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(data, horizon: int) -> pd.DataFrame:
    """Return earned by a trade entered at the next open and exited ``horizon`` days later.

    Value at date t uses prices from t+1 onward, so it must only ever be used as a label.
    """
    o = data.open
    return o.shift(-(1 + horizon)) / o.shift(-1) - 1


def rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame | None = None,
                     min_obs: int = 10) -> pd.Series:
    valid = a.notna() & b.notna()
    if mask is not None:
        valid &= mask
    ra = a.where(valid).rank(axis=1)
    rb = b.where(valid).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(valid.sum(axis=1) >= min_obs)


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame | None = None,
                     q: int = 5) -> pd.DataFrame:
    valid = factor.notna() & fwd.notna()
    if mask is not None:
        valid &= mask
    f = factor.where(valid)
    pct = f.rank(axis=1, pct=True)
    bucket = np.ceil(pct * q).clip(1, q)
    out = {}
    for k in range(1, q + 1):
        out[f"Q{k}"] = fwd.where(bucket == k).mean(axis=1)
    return pd.DataFrame(out)


def factor_report(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame, mask: pd.DataFrame | None,
                  horizon: int, sample_every: int | None = None, q: int = 5) -> pd.DataFrame:
    """Summary table, one row per factor. Uses non-overlapping samples (every ``horizon`` days)."""
    step = sample_every or horizon
    rows = []
    for name, f in factors.items():
        idx = f.index[::step]
        ic = rowwise_spearman(f.loc[idx], fwd.loc[idx], None if mask is None else mask.loc[idx]).dropna()
        qr = quantile_returns(f.loc[idx], fwd.loc[idx], None if mask is None else mask.loc[idx], q).dropna()
        periods_per_year = 252 / horizon
        spread = (qr[f"Q{q}"] - qr["Q1"]) if len(qr) else pd.Series(dtype=float)
        n = len(ic)
        rows.append({
            "factor": name,
            "mean_ic": ic.mean() if n else np.nan,
            "ic_std": ic.std() if n > 1 else np.nan,
            "ic_ir": ic.mean() / ic.std() if n > 1 and ic.std() > 0 else np.nan,
            "t_stat": ic.mean() / ic.std() * np.sqrt(n) if n > 1 and ic.std() > 0 else np.nan,
            "pct_positive": (ic > 0).mean() if n else np.nan,
            "q_spread_ann": spread.mean() * periods_per_year if len(spread) else np.nan,
            "top_q_ann": qr[f"Q{q}"].mean() * periods_per_year if len(qr) else np.nan,
            "n_periods": n,
        })
    return pd.DataFrame(rows).set_index("factor").sort_values("ic_ir", ascending=False)


def ic_decay(factor: pd.DataFrame, data, mask: pd.DataFrame | None, horizons=(1, 5, 10, 21, 63)) -> pd.Series:
    out = {}
    for h in horizons:
        fwd = forward_returns(data, h)
        idx = factor.index[::max(h, 5)]
        out[h] = rowwise_spearman(factor.loc[idx], fwd.loc[idx], None if mask is None else mask.loc[idx]).mean()
    return pd.Series(out, name="mean_ic")


def trailing_ic_weights(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame, mask: pd.DataFrame | None,
                        horizon: int, window: int = 504, min_periods: int = 126) -> pd.DataFrame:
    """Factor weights from *realised* trailing IC.

    The IC at date s needs returns up to s+horizon+1, so it is shifted by horizon+1 rows
    before averaging - weights at t only use ICs that were fully observable at t.
    """
    ics = {n: rowwise_spearman(f, fwd, mask) for n, f in factors.items()}
    ic = pd.DataFrame(ics)
    w = ic.shift(horizon + 1).rolling(window, min_periods=min_periods).mean().clip(lower=0)
    return w
