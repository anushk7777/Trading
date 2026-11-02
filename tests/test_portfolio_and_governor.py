import pandas as pd
import pytest

from indquant.backtest.account import Account
from indquant.config import load_config
from indquant.portfolio import governor
from indquant.portfolio.construction import apply_caps, raw_weights, select_with_buffer


def test_caps_respected_and_excess_redistributed():
    w = pd.Series([0.4, 0.2, 0.2, 0.1, 0.1], index=list("abcde"))
    sec = pd.Series(["x", "x", "y", "z", "z"], index=list("abcde"))
    out = apply_caps(w, max_weight=0.3, sector_cap=0.45, sectors=sec)
    assert out.max() <= 0.3 + 1e-9
    assert out.groupby(sec).sum().max() <= 0.45 + 1e-9
    assert out.sum() == pytest.approx(1.0)


def test_infeasible_caps_leave_cash():
    out = apply_caps(pd.Series([1.0, 1.0, 1.0]), max_weight=0.1)
    assert out.sum() == pytest.approx(0.3)


def test_buffer_keeps_holding_within_band():
    scores = pd.Series({f"S{i}": 100 - i for i in range(20)})
    sel = select_with_buffer(scores, current=["S6"], top_n=5, buffer_mult=1.6)  # S6 ranks 7th <= 8
    assert "S6" in sel and len(sel) == 5
    sel2 = select_with_buffer(scores, current=["S12"], top_n=5, buffer_mult=1.6)  # rank 13 > 8
    assert "S12" not in sel2


def test_inverse_vol_weights_favour_low_vol():
    w = raw_weights(["a", "b"], pd.Series({"a": 0.1, "b": 0.4}), "inverse_vol")
    assert w["a"] == pytest.approx(0.8)


def _gcfg(**kw):
    g = load_config().governor
    for k, v in kw.items():
        g[k] = v
    return g


def test_drawdown_halt_has_hysteresis():
    g = _gcfg(max_drawdown_halt=0.2, resume_drawdown=0.1, profit_lock_activation=10.0)
    acct = Account("x", cash=0, peak_equity=100.0)
    acct.meta["initial_equity"] = 100.0
    assert governor.evaluate(acct, 85, "2024-01-01", g).allow_new_entries
    assert not governor.evaluate(acct, 79, "2024-01-02", g).allow_new_entries
    assert not governor.evaluate(acct, 85, "2024-01-03", g).allow_new_entries  # still halted (dd 15% > 10%)
    assert governor.evaluate(acct, 91, "2024-01-04", g).allow_new_entries


def test_profit_lock_trips_then_rebases():
    g = _gcfg(profit_lock_activation=0.2, profit_lock_trail=0.1, profit_lock_cooldown_days=5,
              max_drawdown_halt=0.9)
    acct = Account("x", cash=0, peak_equity=100.0)
    acct.meta["initial_equity"] = 100.0
    assert governor.evaluate(acct, 130, "2024-01-01", g).exposure_mult == 1.0
    acct.peak_equity = 130
    d = governor.evaluate(acct, 116, "2024-01-02", g)  # floor = 117
    assert d.exposure_mult == 0.5 and acct.meta["lock_trips"] == 1
    assert governor.evaluate(acct, 125, "2024-01-05", g).exposure_mult == 0.5  # cooling off
    assert governor.evaluate(acct, 118, "2024-01-15", g).exposure_mult == 1.0  # rebased, not re-earned yet
    assert acct.meta["lock_base"] == 118


def test_loss_streak_brake_halves_size():
    g = _gcfg(loss_streak=3, loss_streak_size_mult=0.5, loss_streak_cooldown_days=5, profit_lock_activation=10.0)
    acct = Account("x", cash=0, peak_equity=100.0)
    acct.meta.update({"initial_equity": 100.0, "loss_streak": 3})
    assert governor.evaluate(acct, 100, "2024-01-01", g).size_mult == 0.5
    assert governor.evaluate(acct, 100, "2024-01-03", g).size_mult == 0.5
    assert governor.evaluate(acct, 100, "2024-02-01", g).size_mult == 1.0
