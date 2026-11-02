"""Mean-variance portfolio optimizer with transaction costs (the institutional standard).

    maximise   alpha'w  -  lambda * w' Sigma w  -  sum_i c_i |w_i - w0_i|
    subject to 0 <= w_i <= max_weight,  sum(w) <= budget,  sector sums <= sector_cap

* alpha (expected return over the rebalance horizon) follows Grinold's "fundamental law"
  form alpha_i = IC * sigma_i * z_i: the cross-sectional score z is scaled by how much
  the signal has *actually* predicted (trailing realised IC) and by each stock's
  volatility. A signal that stops working shrinks its own alphas towards zero.
* Sigma is a Ledoit-Wolf shrinkage covariance (robust with many stocks, short history).
* The cost term makes the optimizer trade only when the alpha gain beats the charges -
  turnover control falls out of the economics instead of an ad-hoc buffer.
* Cardinality: tiny positions are pruned and the problem is re-solved on the survivors.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def grinold_alpha(z: pd.Series, sigma: pd.Series, ic: float) -> pd.Series:
    return ic * sigma.reindex(z.index) * z


def _solve(alpha, cov, w0, tc, sectors, max_weight, sector_cap, budget, lam, fixed_zero=()):
    import cvxpy as cp

    n = len(alpha)
    w = cp.Variable(n)
    obj = alpha @ w - lam * cp.quad_form(w, cp.psd_wrap(cov)) - cp.sum(cp.multiply(tc, cp.abs(w - w0)))
    cons = [w >= 0, w <= max_weight, cp.sum(w) == budget]  # fully invested; exposure is set by the risk layer
    if sector_cap:
        for name in np.unique(sectors):
            if name == "Unknown":
                continue
            idx = np.nonzero(sectors == name)[0]
            cons.append(cp.sum(w[idx]) <= sector_cap)
    if len(fixed_zero):
        cons.append(w[list(fixed_zero)] == 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    for solver in ("CLARABEL", "OSQP", "SCS"):
        try:
            prob.solve(solver=solver)
        except Exception:  # solver missing / numerical trouble -> next solver
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.clip(np.asarray(w.value).ravel(), 0, None), prob.status
    return None, prob.status


def optimize_weights(alpha: pd.Series, cov: pd.DataFrame, w0: pd.Series, sectors: pd.Series, *,
                     max_weight: float, sector_cap: float | None, budget: float, risk_aversion: float,
                     tcost: float | pd.Series, min_weight: float, max_names: int) -> tuple[pd.Series | None, dict]:
    names = list(alpha.index)
    a = alpha.to_numpy(dtype=float)
    S = cov.reindex(index=names, columns=names).fillna(0.0).to_numpy(dtype=float)
    S = (S + S.T) / 2 + np.eye(len(names)) * 1e-10
    x0 = w0.reindex(names).fillna(0.0).to_numpy(dtype=float)
    tc = (tcost.reindex(names).fillna(tcost.median()).to_numpy(dtype=float) if isinstance(tcost, pd.Series)
          else np.full(len(names), float(tcost)))
    sec = sectors.reindex(names).fillna("Unknown").to_numpy()
    args = dict(alpha=a, cov=S, w0=x0, tc=tc, sectors=sec, max_weight=max_weight, sector_cap=sector_cap,
                budget=budget, lam=risk_aversion)
    try:
        w, status = _solve(**args)
    except ImportError:
        return None, {"status": "cvxpy not installed"}
    if w is None:
        return None, {"status": status}
    # cardinality: prune small weights / keep the largest max_names, then re-solve once
    order = np.argsort(-w)
    keep = set(order[:max_names]) & set(np.nonzero(w >= min_weight)[0])
    zero = [k for k in range(len(names)) if k not in keep]
    if zero and keep:
        w2, status2 = _solve(**args, fixed_zero=zero)
        if w2 is not None:
            w, status = w2, status2
    w[w < 1e-6] = 0.0
    out = pd.Series(w, index=names)
    out = out[out > 0]
    info = {"status": status, "n_names": int((out > 0).sum()), "expected_alpha": float(a @ w),
            "ex_ante_var": float(w @ S @ w), "turnover": float(np.abs(w - x0).sum())}
    return out, info
