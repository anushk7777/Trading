"""The most important test in the repo: future prices must never influence today's decision.

We build signals on the full history and on history truncated at date t, then also
*corrupt* every price after t. Scores at t, and the whole simulated account up to t, must
be identical. This covers features, factor ranks, walk-forward ML (purging/embargo),
eligibility filters, the engine and both strategies.
"""
import numpy as np
import pandas as pd

from indquant.backtest.runner import run_backtest
from indquant.config import load_config
from indquant.data.base import MarketData
from indquant.strategies.factor_portfolio import FactorPortfolioStrategy, rebalance_flags
from indquant.strategies.swing import SwingStrategy

from .conftest import small_overrides


def _corrupt_after(data: MarketData, t: pd.Timestamp, seed: int = 1) -> MarketData:
    rng = np.random.default_rng(seed)
    after = data.dates > t
    scale = pd.DataFrame(np.exp(rng.normal(0, 0.3, data.close.shape)), index=data.dates, columns=data.symbols)
    keep = np.broadcast_to(~np.asarray(after)[:, None], data.close.shape)

    def bad(df):
        return df.where(keep, df * scale)
    return MarketData(open=bad(data.open), high=bad(data.high), low=bad(data.low),
                      close=bad(data.close), volume=bad(data.volume), benchmark=data.benchmark.where(~after, 1.0),
                      index=data.index.where(~after, data.index * 3), sectors=data.sectors, source="synthetic")


def test_scores_at_t_ignore_the_future(small_data, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path))
    flags = rebalance_flags(small_data.dates)
    flag_dates = small_data.dates[flags]
    t = flag_dates[flag_dates > small_data.dates[600]][0]
    variants = {"full": small_data, "truncated": small_data.truncate(t), "corrupted": _corrupt_after(small_data, t)}
    core, swing = {}, {}
    for k, d in variants.items():
        c = FactorPortfolioStrategy(cfg)
        c.prepare(d)
        s = SwingStrategy(cfg)
        s.prepare(d)
        core[k], swing[k] = c, s
    assert core["full"].ml_model.fit_dates, "ML should be active in this test"
    assert core["full"].ml_score.loc[t].notna().sum() > 10
    ref_core = core["truncated"].score.loc[t]
    ref_prob = swing["truncated"].expected_r.loc[t]
    for k in ("full", "corrupted"):
        pd.testing.assert_series_equal(core[k].score.loc[t], ref_core, check_exact=False, rtol=1e-9, atol=1e-12)
        pd.testing.assert_series_equal(swing[k].expected_r.loc[t], ref_prob, check_exact=False, rtol=1e-9, atol=1e-12)
        pd.testing.assert_series_equal(swing[k].rule_score.loc[t], swing["truncated"].rule_score.loc[t])


def test_simulated_account_up_to_t_ignores_the_future(small_data, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    t = small_data.dates[700]
    a = run_backtest(cfg, small_data.truncate(t))
    b = run_backtest(cfg, _corrupt_after(small_data, t))
    eq_a = a.equity["total"]
    eq_b = b.equity["total"].loc[:t]
    pd.testing.assert_series_equal(eq_a, eq_b, check_exact=False, rtol=1e-10)
    ta = a.trades.drop(columns=["tax_lots"])
    tb = b.trades[b.trades["date"] <= t].drop(columns=["tax_lots"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ta.reset_index(drop=True), tb, check_exact=False, rtol=1e-10)
