import numpy as np
import pandas as pd
import pytest

from indquant.backtest.account import Account, Order
from indquant.backtest.costs import CostModel
from indquant.backtest.engine import Engine
from indquant.data.base import MarketData


def _market(o, h, l, c, v=None):
    idx = pd.bdate_range("2021-01-01", periods=len(c))
    f = lambda x: pd.DataFrame({"S": x}, index=idx)  # noqa: E731
    v = v or [1e7] * len(c)
    return MarketData(open=f(o), high=f(h), low=f(l), close=f(c), volume=f(v),
                      benchmark=pd.Series(100.0, index=idx), index=pd.Series(100.0, index=idx))


class Scripted:
    """Emits pre-defined orders at given bars."""

    name = "test"

    def __init__(self, script):
        self.script = script

    def prepare(self, data):
        pass

    def on_bar(self, i, ctx):
        return [Order(**kw, created=str(ctx.date.date())) for kw in self.script.get(i, [])]


NO_SLIP = CostModel(slippage_bps=0, impact_coef=0)


def test_order_fills_next_open_with_costs_and_integer_shares():
    d = _market([100, 110, 120], [100, 111, 121], [100, 109, 119], [100, 110, 120])
    eng = Engine(d, NO_SLIP)
    acct = Account("t", cash=10_000)
    eng.run(Scripted({0: [dict(symbol="S", shares=50, reason="t")]}), acct, 0)
    buy = acct.trades[0]
    assert buy["date"] == d.dates[1] and buy["price"] == 110.0  # next open, not the signal close
    assert buy["shares"] == 50
    assert acct.cash == pytest.approx(10_000 - 5_500 - NO_SLIP.cost("buy", 5_500))
    assert acct.history[-1]["equity"] == pytest.approx(acct.cash + 50 * 120)


def test_buy_is_limited_by_cash():
    d = _market([100, 100], [100, 100], [100, 100], [100, 100])
    acct = Account("t", cash=1_000)
    Engine(d, NO_SLIP).run(Scripted({0: [dict(symbol="S", shares=50, reason="t")]}), acct, 0)
    assert acct.positions["S"].shares == 9  # 10 shares + charges would exceed cash
    assert acct.cash >= 0


def test_stop_assumed_before_target_on_ambiguous_bar():
    d = _market([100, 100, 100], [100, 100, 106], [100, 100, 94], [100, 100, 100])
    acct = Account("t", cash=100_000)
    s = Scripted({0: [dict(symbol="S", shares=10, reason="t", stop_dist=3.0, target_dist=5.0)]})
    Engine(d, NO_SLIP).run(s, acct, 0)
    sell = [t for t in acct.trades if t["side"] == "sell"][0]
    assert sell["reason"] == "stop" and sell["price"] == pytest.approx(97.0)


def test_gap_through_stop_fills_at_open():
    d = _market([100, 100, 90], [100, 100, 92], [100, 100, 89], [100, 100, 91])
    acct = Account("t", cash=100_000)
    s = Scripted({0: [dict(symbol="S", shares=10, reason="t", stop_dist=3.0, target_dist=5.0)]})
    Engine(d, NO_SLIP).run(s, acct, 0)
    sell = [t for t in acct.trades if t["side"] == "sell"][0]
    assert sell["reason"] == "stop_gap" and sell["price"] == 90.0


def test_fifo_realised_pnl_and_loss_streak_meta():
    d = _market([100, 100, 100, 80, 80], [100] * 3 + [80, 80], [100] * 3 + [80, 80], [100] * 3 + [80, 80])
    acct = Account("t", cash=100_000)
    s = Scripted({0: [dict(symbol="S", shares=10, reason="t")], 2: [dict(symbol="S", shares=-10, reason="x")]})
    Engine(d, NO_SLIP).run(s, acct, 0)
    sell = [t for t in acct.trades if t["side"] == "sell"][0]
    cost_basis = 1000 + NO_SLIP.cost("buy", 1000)
    assert sell["pnl"] == pytest.approx(800 - NO_SLIP.cost("sell", 800) - cost_basis)
    assert acct.meta["loss_streak"] == 1 and acct.meta["last_exit"]["S"]["pnl"] < 0
    assert "S" not in acct.positions


def test_untradable_bar_carries_order_until_ttl_expires():
    d = _market([100, np.nan, np.nan, 100], [100, np.nan, np.nan, 100], [100, np.nan, np.nan, 100],
                [100, np.nan, np.nan, 100])
    acct = Account("t", cash=100_000)
    Engine(d, NO_SLIP).run(Scripted({0: [dict(symbol="S", shares=5, reason="t", ttl=3)]}), acct, 0)
    assert acct.trades and acct.trades[0]["date"] == d.dates[3]
