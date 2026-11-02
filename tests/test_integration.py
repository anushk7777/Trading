"""Sanity checks against known ground truth from the synthetic market.

* With planted persistent drift, momentum should show a significant positive IC.
* In a pure random walk there is nothing to find: the strategy must NOT show significant
  alpha. If it did, the pipeline would be leaking information or overfitting.
"""
import pytest

from indquant.backtest.runner import run_backtest
from indquant.config import load_config
from indquant.data import generate_synthetic_market
from indquant.data.universe import eligibility_mask
from indquant.features.factors import compute_factors
from indquant.research.factor_analysis import factor_report, forward_returns

from .conftest import small_overrides


@pytest.fixture(scope="module")
def planted():
    return generate_synthetic_market(n_stocks=120, n_days=1500, seed=11, alpha_strength=2.0)


@pytest.fixture(scope="module")
def noise():
    return generate_synthetic_market(n_stocks=120, n_days=1500, seed=11, alpha_strength=0.0)


def _cfg(tmp_path, **kw):
    o = small_overrides(tmp_path, **{"ml.enabled": False, "swing.enabled": False, "allocation.core": 1.0,
                                     "allocation.swing": 0.0})
    o.update(kw)
    return load_config(overrides=o)


def test_planted_momentum_is_detected(planted, tmp_path):
    cfg = _cfg(tmp_path)
    mask = eligibility_mask(planted, cfg)
    rep = factor_report(compute_factors(planted), forward_returns(planted, 21), mask, 21)
    assert rep.loc["mom_12_1", "mean_ic"] > 0.02
    assert rep.loc["mom_12_1", "t_stat"] > 2


def test_planted_alpha_beats_random_walk_version(planted, noise, tmp_path):
    cfg = _cfg(tmp_path)
    alpha_p = run_backtest(cfg, planted).metrics["total"]["alpha"]
    alpha_n = run_backtest(cfg, noise).metrics["total"]["alpha"]
    assert alpha_p > alpha_n


def test_no_significant_alpha_in_pure_noise(noise, tmp_path):
    res = run_backtest(_cfg(tmp_path), noise)
    m = res.metrics["total"]
    assert abs(m["alpha_tstat"]) < 2.0, m
    assert any("NOT STATISTICALLY SIGNIFICANT" in w or "UNDERPERFORMS" in w for w in res.warnings)


def test_ml_pipeline_runs_on_noise_without_finding_edge(noise, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"swing.enabled": False, "allocation.core": 1.0,
                                                            "allocation.swing": 0.0}))
    res = run_backtest(cfg, noise)
    assert res.strategies["core"].ml_model.fit_dates
    assert abs(res.metrics["total"]["alpha_tstat"]) < 2.5
