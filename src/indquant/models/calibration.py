"""Calibrated fractional-Kelly sizing from the strategy's own realised edge.

A model's raw "expected R" is a good *ranking* but usually a poor *size* estimate (testing
showed the predicted magnitude carries little information even when the ranking works).
So Kelly is applied to what matters: the realised R-multiple, after costs, of the trades
the strategy actually *selects* (prediction above the entry threshold), using only trades
whose outcome was already known at the decision date:

    risk per trade = min(fixed rule, fraction x mean_R / var_R)

While the selected trades keep paying, the size stays at the fixed rule; when their
realised edge fades towards zero, Kelly shrinks the size automatically - a strategy whose
edge disappears sizes itself out of the market instead of bleeding costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def selected_edge(selected: pd.DataFrame, realised_net_r: pd.DataFrame, horizon: int, window: int = 504,
                  min_obs: int = 300) -> pd.DataFrame:
    """Per-date mean / variance of realised net R over past *selected* trades with known outcomes."""
    y = realised_net_r.where(selected.fillna(False).astype(bool))
    v = y.notna()
    sums = pd.DataFrame({"n": v.sum(axis=1), "sy": y.sum(axis=1), "syy": (y * y).sum(axis=1)})
    known = sums.shift(horizon + 1).rolling(window, min_periods=1).sum()   # outcome of s known at s+1+horizon
    n = known["n"].replace(0, np.nan)
    mean = known["sy"] / n
    var = known["syy"] / n - mean ** 2
    out = pd.DataFrame({"mean": mean, "var": var, "n": known["n"]})
    out.loc[known["n"] < min_obs, ["mean", "var"]] = np.nan
    return out


def kelly_risk(mu_net: float, var: float, fraction: float, base_risk: float) -> float:
    """Fraction of equity to risk on one trade: min(base rule, fraction x Kelly)."""
    if mu_net is None or var is None or np.isnan(mu_net) or np.isnan(var) or var <= 0:
        return base_risk
    return float(min(base_risk, max(mu_net, 0.0) / var * fraction))
