"""PBO, factor attribution and stress tests."""
import numpy as np
import pandas as pd
import pytest

from indquant.backtest.attribution import regress
from indquant.backtest.runner import run_backtest
from indquant.backtest.stress import historical, hypothetical
from indquant.config import load_config
from indquant.research.overfitting import probability_of_backtest_overfitting

from .conftest import small_overrides


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1000, 20)), index=idx)
    assert probability_of_backtest_overfitting(noise)["pbo"] > 0.3
    edge = noise.copy()
    edge[0] += 0.002  # one configuration genuinely better everywhere
    assert probability_of_backtest_overfitting(edge)["pbo"] < 0.05
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(noise[[0]])


def test_attribution_recovers_known_exposures():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=800)
    f = pd.DataFrame(rng.normal(0, 0.01, (800, 3)), index=idx, columns=["MKT", "MOM", "SMB"])
    y = 0.9 * f["MKT"] + 0.5 * f["MOM"] + rng.normal(0, 0.002, 800)
    out = regress(y, f)
    t = out["table"]
    assert t.loc["MKT", "coef"] == pytest.approx(0.9, abs=0.05) and t.loc["MOM", "coef"] == pytest.approx(0.5, abs=0.05)
    assert abs(out["alpha_t"]) < 2.5 and out["r2"] > 0.9


def test_stress_windows_and_shocks():
    idx = pd.bdate_range("2019-06-01", "2020-06-30")
    eq = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    h = historical(eq, eq * 0.9)
    assert list(h["event"]) == ["COVID-19 crash"]
    w = pd.Series({"A": 0.5, "B": 0.3})
    hyp = hypothetical(w, pd.Series({"A": 1.2, "B": 0.8}), pd.Series({"A": "IT", "B": "Bank"}), 1_000_000)
    assert hyp.loc[0, "portfolio_return"] == pytest.approx(-0.1 * (0.5 * 1.2 + 0.3 * 0.8))
    assert any("IT" in s for s in hyp["scenario"])


def test_backtest_includes_attribution_and_stress(small_data, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False, "swing.enabled": False,
                                                            "allocation.core": 1.0, "allocation.swing": 0.0}))
    res = run_backtest(cfg, small_data)
    assert "MKT" in res.attribution["table"].index and "MOM" in res.attribution["table"].index
    assert "factor_alpha_tstat" in res.metrics["total"]
    assert {"historical", "hypothetical"} <= set(res.stress)
    assert len(res.stress["hypothetical"]) >= 4
