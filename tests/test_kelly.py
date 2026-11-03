"""Calibrated fractional Kelly: honest calibration, point-in-time, can only shrink positions."""
import numpy as np
import pandas as pd
import pytest

from indquant.backtest.account import Account
from indquant.backtest.costs import CostModel
from indquant.backtest.engine import BarContext, Engine
from indquant.config import load_config
from indquant.models.calibration import kelly_risk, selected_edge
from indquant.strategies.swing import SwingStrategy

from .conftest import small_overrides


def test_selected_edge_statistics():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    realised = pd.DataFrame(rng.normal(0.2, 1.0, (300, 20)), index=idx)
    chosen = pd.DataFrame(False, index=idx, columns=realised.columns)
    chosen.iloc[:, :3] = True                                # three trades a day
    e = selected_edge(chosen, realised, horizon=10, window=200, min_obs=100).iloc[-1]
    assert e["mean"] == pytest.approx(0.2, abs=0.1) and e["var"] == pytest.approx(1.0, abs=0.15)
    assert e["n"] == 600                                     # 200 known days x 3 trades


def test_selected_edge_never_uses_unknown_outcomes():
    idx = pd.bdate_range("2020-01-01", periods=300)
    realised = pd.DataFrame(-1.0, index=idx, columns=range(5))
    realised.iloc[150:] = 5.0                                # spectacular results only in the future
    chosen = pd.DataFrame(True, index=idx, columns=realised.columns)
    e = selected_edge(chosen, realised, horizon=10, window=500, min_obs=10)
    assert e["mean"].iloc[155] == pytest.approx(-1.0)        # day 155 only knows outcomes of days <= 144
    assert e["mean"].iloc[-1] > 0


def test_kelly_risk_only_shrinks():
    assert kelly_risk(0.02, 1.2, 0.25, 0.0075) == pytest.approx(0.02 / 1.2 * 0.25)
    assert kelly_risk(0.50, 1.0, 0.25, 0.0075) == 0.0075     # capped at the fixed rule
    assert kelly_risk(-0.1, 1.0, 0.25, 0.0075) == 0.0
    assert kelly_risk(float("nan"), 1.0, 0.25, 0.0075) == 0.0075


def test_swing_orders_never_larger_with_kelly(small_data, tmp_path):
    base = load_config(overrides=small_overrides(tmp_path, **{"swing.kelly_fraction": None,
                                                             "swing.kelly_min_obs": 50}))
    kel = load_config(overrides=small_overrides(tmp_path, **{"swing.kelly_fraction": 0.25,
                                                            "swing.kelly_min_obs": 50}))
    s0, s1 = SwingStrategy(base), SwingStrategy(kel)
    s0.prepare(small_data)
    s1.prepare(small_data)
    assert s1.calibration is not None and s1.calibration["mean"].notna().any()
    eng = Engine(small_data, CostModel())
    compared = 0
    for i in range(700, 880, 5):
        o0 = {o.symbol: o.shares for o in s0.on_bar(i, BarContext(eng, i, Account("swing", 300_000), 300_000))}
        o1 = {o.symbol: o.shares for o in s1.on_bar(i, BarContext(eng, i, Account("swing", 300_000), 300_000))}
        for sym in set(o0) & set(o1):
            assert o1[sym] <= o0[sym]
            compared += 1
    assert compared > 0 or s1.rejections.get("kelly_edge_too_small", 0) > 0
