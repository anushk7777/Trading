"""Corporate actions in books held across days (live desk and paper)."""
import pandas as pd
import pytest

from indquant.backtest.account import Account, Lot, Position
from indquant.config import load_config
from indquant.data.base import MarketData
from indquant.live.corporate_actions import reconcile
from indquant.live.desk import DeskStore, TradeDesk
from indquant.live.signals import step_accounts

from .conftest import small_overrides
from .test_desk import FakeReviewer


def _book(price=2990.0, day="2024-01-02"):
    acct = Account("swing", cash=1000.0)
    acct.positions["X"] = Position("X", 13, [Lot(day, 13, 2993.0)], day, 2990.0, stop=2800.0, target=3200.0,
                                   atr=50.0, last_price=price, last_date=day)
    return acct


def _data(close_on_day):
    idx = pd.bdate_range("2024-01-01", periods=3)
    c = pd.DataFrame({"X": [1.0, close_on_day, close_on_day]}, index=idx)
    return MarketData(open=c, high=c, low=c, close=c, volume=c, benchmark=c["X"], index=c["X"])


def test_bonus_rescales_shares_cost_and_levels():
    acct = _book()
    ev = reconcile(acct, _data(1495.0))
    pos = acct.positions["X"]
    assert ev[0]["kind"] == "split" and ev[0]["multiplier"] == "2"
    assert pos.shares == 26 and pos.lots[0].cost == pytest.approx(1496.5)
    assert pos.stop == pytest.approx(1400) and pos.target == pytest.approx(1600) and pos.atr == pytest.approx(25)
    assert pos.shares * pos.last_price == pytest.approx(13 * 2990)  # value unchanged
    assert acct.cash == 1000.0


def test_dividend_is_credited_and_noop_when_unchanged():
    acct = _book()
    ev = reconcile(acct, _data(2990.0 * 0.99))
    assert ev[0]["kind"] == "dividend" and acct.cash == pytest.approx(1000 + 13 * 29.9)
    assert acct.positions["X"].shares == 13
    acct2 = _book()
    assert reconcile(acct2, _data(2990.0)) == [] and acct2.cash == 1000.0


def _halved(data, sym):
    f = lambda df, m: df.assign(**{sym: df[sym] * m})  # noqa: E731
    return MarketData(open=f(data.open, .5), high=f(data.high, .5), low=f(data.low, .5), close=f(data.close, .5),
                      volume=f(data.volume, 2), benchmark=data.benchmark, index=data.index, sectors=data.sectors,
                      source="synthetic")


def test_desk_applies_bonus_and_reprotects(tmp_path, small_data):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    desk = TradeDesk(cfg, store=DeskStore(tmp_path / "d.db"), reviewer=FakeReviewer())
    desk.init(1_000_000)
    d0 = small_data.truncate(small_data.dates[500])
    desk.refresh(d0)
    p = next(p for p in desk.store.proposals("status='pending'") if p["sleeve"] == "swing")
    desk.approve(p["id"])
    desk.confirm_fill(p["id"], p["qty"], p["limit_price"])
    desk.refresh(d0)  # marks the position at the data date
    before = desk.store.accounts()["swing"].positions[p["symbol"]]
    res = desk.refresh(_halved(small_data, p["symbol"]).truncate(small_data.dates[501]))
    after = desk.store.accounts()["swing"].positions[p["symbol"]]
    assert after.shares == before.shares * 2 and after.stop == pytest.approx(before.stop / 2)
    assert any("split/bonus" in a for a in res["alerts"]) and any("GTT" in a for a in res["alerts"])


def test_paper_book_applies_bonus(tmp_path, small_data):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    r1 = step_accounts(cfg, small_data.truncate(small_data.dates[500]), None, 1_000_000)
    r2 = step_accounts(cfg, small_data.truncate(small_data.dates[501]), r1.accounts, 1_000_000)
    acct = r2.accounts["core"]
    sym = next(iter(acct.positions))
    shares = acct.positions[sym].shares
    eq_before = acct.equity(small_data.close.iloc[502])
    r3 = step_accounts(cfg, _halved(small_data, sym).truncate(small_data.dates[502]), r2.accounts, 1_000_000)
    assert any(e["symbol"] == sym and e["kind"] == "split" for e in r3.corporate_actions)
    assert r3.accounts["core"].positions[sym].shares >= shares * 2 - 1
    eq_after = r3.accounts["core"].equity(_halved(small_data, sym).close.iloc[502])
    assert eq_after == pytest.approx(eq_before, rel=0.02)
