import pytest

from indquant.backtest.costs import CostModel
from indquant.config import load_config


def test_delivery_buy_charges_match_hand_calculation():
    cm = CostModel.from_config(load_config().costs)
    b = cm.breakdown("buy", 100_000)
    assert b.stt == pytest.approx(100.0)
    assert b.stamp == pytest.approx(15.0)
    assert b.exchange == pytest.approx(2.97)
    assert b.sebi == pytest.approx(0.10)
    assert b.gst == pytest.approx(0.18 * (2.97 + 0.10))
    assert b.dp == 0
    assert b.total == pytest.approx(100 + 15 + 2.97 + 0.10 + 0.18 * 3.07)


def test_delivery_sell_charges_include_dp_and_no_stamp():
    cm = CostModel.from_config(load_config().costs)
    s = cm.breakdown("sell", 100_000)
    assert s.stamp == 0
    assert s.dp == pytest.approx(15.93)
    assert s.total == pytest.approx(100 + 2.97 + 0.10 + 0.18 * 3.07 + 15.93)


def test_brokerage_is_capped_per_order():
    cm = CostModel(brokerage_pct=0.0003, brokerage_max_per_order=20)
    assert cm.breakdown("buy", 10_000_000).brokerage == 20
    assert cm.breakdown("buy", 10_000).brokerage == pytest.approx(3.0)


def test_slippage_grows_with_order_size_and_is_capped():
    cm = CostModel(slippage_bps=5, impact_coef=0.1, max_slippage=0.02)
    small = cm.slippage(1e5, adv_value=1e9, daily_vol=0.02)
    big = cm.slippage(1e8, adv_value=1e9, daily_vol=0.02)
    assert 5e-4 <= small < big <= 0.02
    assert cm.slippage(1e12, adv_value=1e6, daily_vol=0.05) == 0.02
