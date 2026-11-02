"""Mean-variance optimizer with costs."""
import numpy as np
import pandas as pd
import pytest

from indquant.portfolio.optimizer import optimize_weights


def _inputs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"S{i}" for i in range(n)]
    alpha = pd.Series(rng.normal(0, 0.01, n), index=names)
    f = rng.normal(0, 0.05, (n, 3))
    cov = pd.DataFrame(f @ f.T + np.diag(rng.uniform(0.002, 0.01, n)), index=names, columns=names)
    sectors = pd.Series(["A"] * 10 + ["B"] * 10 + ["Unknown"] * 10, index=names)
    return alpha, cov, sectors


KW = dict(max_weight=0.1, sector_cap=0.3, budget=0.99, risk_aversion=1.0, tcost=0.002, min_weight=0.01, max_names=40)


def test_constraints_hold():
    alpha, cov, sectors = _inputs()
    w, info = optimize_weights(alpha, cov, pd.Series(dtype=float), sectors, **KW)
    assert info["status"].startswith("optimal")
    assert w.sum() == pytest.approx(0.99, abs=1e-4) and (w >= 0).all() and w.max() <= 0.1 + 1e-6
    by_sector = w.groupby(sectors.reindex(w.index)).sum()
    assert by_sector.get("A", 0) <= 0.3 + 1e-6 and by_sector.get("B", 0) <= 0.3 + 1e-6
    assert (w >= 0.01 - 1e-6).all() and alpha.reindex(w.index).mean() > alpha.mean()


def test_costs_suppress_pointless_trades():
    alpha, cov, sectors = _inputs()
    w_free, _ = optimize_weights(alpha, cov, pd.Series(dtype=float), sectors, **KW)
    flat = alpha * 0 + 1e-6                                  # no real alpha differences
    kw = {**KW, "tcost": 0.05}
    w2, info = optimize_weights(flat, cov, w_free, sectors, **kw)
    assert info["turnover"] < 0.05                           # keeps the existing book instead of churning


def test_missing_solver_returns_none(monkeypatch):
    import indquant.portfolio.optimizer as opt

    def boom(**kw):
        raise ImportError("no cvxpy")

    monkeypatch.setattr(opt, "_solve", boom)
    alpha, cov, sectors = _inputs()
    w, info = opt.optimize_weights(alpha, cov, pd.Series(dtype=float), sectors, **KW)
    assert w is None and "cvxpy" in info["status"]
