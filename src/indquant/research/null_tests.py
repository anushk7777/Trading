"""Hypothesis test: is the strategy better than *random* stock picking under identical rules?

The null hypothesis is "the ranking signal has no information". We keep everything else
identical - universe, position sizing, rebalancing schedule, risk controls, costs - and
replace the signal with random numbers. If many random portfolios match the strategy's
Sharpe, its performance came from the rules/market exposure, not from the signal.
"""
from __future__ import annotations

import copy
import logging
from collections import Counter

import numpy as np
import pandas as pd

from ..backtest.account import Account
from ..backtest.costs import CostModel
from ..backtest.engine import Engine, history_frame
from ..backtest.metrics import returns_from_equity, sharpe
from ..backtest.runner import start_index

log = logging.getLogger(__name__)


def _randomised(strategy, rng: np.random.Generator):
    s = copy.copy(strategy)
    if hasattr(s, "diagnostics"):
        s.diagnostics = []
    if hasattr(s, "rejections"):
        s.rejections = Counter()
    mask = s.mask
    noise = pd.DataFrame(rng.random(mask.shape), index=mask.index, columns=mask.columns).where(mask)
    if s.name == "core":
        s.score = noise
    else:
        if getattr(s, "expected_r", None) is not None:
            # random "expected R" with the same cross-sectional spread as a real model output
            s.expected_r = (noise - 0.5) * 0.6
        s.rule_score = noise.where(s.rule_score.notna())
    return s


def random_signal_test(cfg, data, strategy, actual_equity: pd.Series, n_runs: int = 20, seed: int = 0) -> dict:
    """Return the null Sharpe distribution and a one-sided p-value for ``strategy``."""
    engine = Engine(data, CostModel.from_config(cfg.costs), cash_yield=float(cfg.risk.cash_yield))
    i0 = start_index(data, cfg)
    rf = float(cfg.risk.risk_free_rate)
    actual = sharpe(returns_from_equity(actual_equity), rf)
    rng = np.random.default_rng(seed)
    capital = float(cfg.general.initial_capital) * float(cfg.allocation[strategy.name])
    null = []
    for k in range(n_runs):
        strat = _randomised(strategy, rng)
        acct = Account(name=strategy.name, cash=capital, peak_equity=capital)
        engine.run(strat, acct, i0)
        null.append(sharpe(returns_from_equity(history_frame(acct)["equity"]), rf))
        log.info("null run %d/%d: Sharpe %.2f", k + 1, n_runs, null[-1])
    null_arr = np.array(null)
    p = (1 + np.sum(null_arr >= actual)) / (n_runs + 1)
    return {"sleeve": strategy.name, "actual_sharpe": actual, "null_mean": float(np.mean(null_arr)),
            "null_95pct": float(np.quantile(null_arr, 0.95)), "p_value": float(p), "n_runs": n_runs,
            "null_sharpes": null_arr.tolist()}
