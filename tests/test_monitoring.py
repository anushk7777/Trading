"""Transaction-cost analysis and live-vs-backtest drift monitoring."""
import numpy as np
import pandas as pd
import pytest

from indquant.config import load_config
from indquant.data.base import MarketData
from indquant.live.desk import DeskStore, TradeDesk
from indquant.live.tca import shortfall_table, summary
from indquant.research.expectations import build, check, load, save

from .conftest import small_overrides
from .test_desk import FakeReviewer


def test_shortfall_signs_and_summary():
    idx = pd.bdate_range("2024-01-01", periods=3)
    o = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx)
    data = MarketData(open=o, high=o, low=o, close=o, volume=o, benchmark=o["A"], index=o["A"])
    fills = pd.DataFrame({"data_date": [str(idx[0].date())] * 2, "symbol": ["A", "A"], "side": ["BUY", "SELL"],
                          "ref_price": [100.0, 100.0], "fill_price": [101.101, 100.899]})
    t = shortfall_table(fills, data)
    assert t["vs_open_bps"].tolist() == pytest.approx([10.0, 10.0], abs=0.01)   # both cost 10 bps vs next open
    assert t["vs_signal_close_bps"].iloc[0] == pytest.approx(110.1, abs=0.1)
    assert "need 20+" in summary(t, 5.0)["message"]
    big = pd.concat([t] * 15, ignore_index=True)
    assert "slippage_bps" in summary(big, 5.0)["message"]


def test_drift_check_flags_underperformance():
    rng = np.random.default_rng(0)
    bt = pd.Series(rng.normal(0.0006, 0.01, 1500))
    exp = {"quantiles": build(bt, reps=500)}
    idx = pd.bdate_range("2024-01-01", periods=80)
    good = pd.Series(np.cumprod(1 + rng.normal(0.0006, 0.01, 80)), index=idx)
    bad = pd.Series(np.cumprod(np.full(80, 1 - 0.004)), index=idx)
    assert check(good, exp)["status"] in ("normal", "above")
    out = check(bad, exp)
    assert out["status"] == "below" and out["horizon"] == 63 and "DRIFT ALERT" in out["message"]
    assert check(good.iloc[:10], exp)["status"] == "insufficient"


def test_expectations_roundtrip_and_desk_state(tmp_path, small_data):
    save(tmp_path / "e.json", {"21": {"p05": -0.1, "p25": 0, "p50": 0.01, "p75": 0.02, "p95": 0.1}}, {"source": "x"})
    assert load(tmp_path / "e.json")["quantiles"]["21"]["p05"] == -0.1
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    desk = TradeDesk(cfg, store=DeskStore(tmp_path / "d.db"), reviewer=FakeReviewer())
    desk.init(1_000_000)
    desk.refresh(small_data.truncate(small_data.dates[500]))
    p = desk.store.proposals("status='pending'")[0]
    desk.approve(p["id"])
    desk.confirm_fill(p["id"], p["qty"], p["limit_price"] * 1.001)
    st = desk.state()
    assert st["tca"]["n"] == 1 and st["drift"]["status"] == "insufficient"
    table, summ = desk.tca(small_data)
    assert "vs_open_bps" in table and summ["basis"] == "vs_open_bps"
