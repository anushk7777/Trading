"""Factor attribution: how much of the return is genuine stock selection?

Strategy returns are regressed on the market and on long-short factor portfolios built
from the same universe (momentum, low volatility, size, and value/quality when
fundamentals exist). If the "alpha" disappears once these exposures are accounted for,
the strategy is mostly harvesting known factor premia - which can often be bought more
cheaply through a factor index fund.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.preprocessing import cs_rank

FACTOR_SPECS = {"MOM": "mom_12_1", "LOWVOL": "low_vol", "VALUE": "value", "QUALITY": "quality"}


def long_short_returns(signal: pd.DataFrame, rets: pd.DataFrame, mask: pd.DataFrame, flags: np.ndarray,
                       q: float = 1 / 3) -> pd.Series:
    """Equal-weight top-minus-bottom portfolio, formed on rebalance days, earning from the next day."""
    r = cs_rank(signal, mask)
    long = (r >= 0.5 - q).astype(float).where(r.notna(), 0.0)
    short = (r <= -0.5 + q).astype(float).where(r.notna(), 0.0)
    wl = long.div(long.sum(axis=1).replace(0, np.nan), axis=0)
    ws = short.div(short.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).where(pd.Series(flags, index=signal.index), np.nan).ffill().fillna(0.0)
    return (w.shift(1) * rets.fillna(0.0)).sum(axis=1)


def factor_returns(data, mask: pd.DataFrame, factors: dict[str, pd.DataFrame], flags: np.ndarray) -> pd.DataFrame:
    rets = data.close.pct_change(fill_method=None)
    out = {"MKT": data.benchmark.pct_change(fill_method=None)}
    for label, name in FACTOR_SPECS.items():
        if name in factors:
            out[label] = long_short_returns(factors[name], rets, mask, flags)
    size = (data.close * data.volume).rolling(63, min_periods=20).mean()
    out["SMB"] = long_short_returns(-size, rets, mask, flags)  # small minus big (traded value as size proxy)
    return pd.DataFrame(out)


def regress(strategy: pd.Series, factors: pd.DataFrame, rf: float = 0.0) -> dict:
    d_rf = (1 + rf) ** (1 / 252) - 1
    df = pd.concat([strategy.rename("y"), factors], axis=1, join="inner").dropna()
    df = df.iloc[1:] if len(df) else df
    if len(df) < 60:
        return {}
    y = df["y"].to_numpy() - d_rf
    X = df.drop(columns="y").copy()
    X["MKT"] = X["MKT"] - d_rf
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    dof = max(len(y) - Xm.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(Xm.T @ Xm)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tstat = np.where(se > 0, beta / se, np.nan)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    names = ["alpha"] + list(X.columns)
    table = pd.DataFrame({"coef": beta, "t_stat": tstat}, index=names)
    table.loc["alpha", "coef"] *= 252  # annualised
    return {"table": table, "alpha": float(table.loc["alpha", "coef"]), "alpha_t": float(table.loc["alpha", "t_stat"]),
            "r2": float(r2), "n_obs": int(len(y))}


def attribute(equity: pd.Series, data, cfg) -> dict:
    from ..data.universe import eligibility_mask
    from ..features.factors import compute_factors
    from ..strategies.factor_portfolio import rebalance_flags

    mask = eligibility_mask(data, cfg)
    fr = factor_returns(data, mask, compute_factors(data), rebalance_flags(data.dates))
    return regress(equity.pct_change(fill_method=None), fr.reindex(equity.index), float(cfg.risk.risk_free_rate))
