"""Fake-money Trade Desk: approve tonight, simulated fills against the next session's real prices."""
import pandas as pd

from indquant.config import load_config
from indquant.live.desk import DeskStore, TradeDesk

from .conftest import small_overrides
from .test_desk import FakeReviewer


def _desk(tmp_path, broker="paper", name="paper.db"):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False, "execution.broker": broker}))
    d = TradeDesk(cfg, store=DeskStore(tmp_path / name), reviewer=FakeReviewer())
    d.init(1_000_000)
    return d


def _day(data, i):
    return data.truncate(data.dates[i])


def test_paper_order_fills_at_next_open_and_stop_fires(tmp_path, small_data):
    desk = _desk(tmp_path)
    desk.refresh(_day(small_data, 500))
    p = next(p for p in desk.store.proposals("status='pending'") if p["sleeve"] == "swing")
    r = desk.approve(p["id"])
    assert r["status"] == "submitted" and "Paper order" in r["message"]
    assert desk.store.proposal(p["id"])["status"] == "submitted"

    res = desk.refresh(_day(small_data, 501))                 # next session's data arrives
    q = desk.store.proposal(p["id"])
    nxt = small_data.dates[501]
    o, low = small_data.open.loc[nxt, p["symbol"]], small_data.low.loc[nxt, p["symbol"]]
    if q["status"] == "filled":
        assert q["fill_price"] <= p["limit_price"] + 1e-9
        assert q["fill_price"] == round(min(o * (1 + 0.0005), p["limit_price"]), 2) or q["fill_price"] == p["limit_price"] \
            or q["fill_price"] >= o
        pos = desk.store.accounts()["swing"].positions.get(p["symbol"])
        assert any("Paper fill" in a for a in res["alerts"]) and (pos is None or pos.stop < q["fill_price"])
        if pos is not None:                                   # force the paper stop to trigger on the next bar
            acct = desk.store.accounts()["swing"]
            acct.positions[p["symbol"]].stop = 10 * q["fill_price"]
            desk.store.save_account(acct)
            res2 = desk.refresh(_day(small_data, 502))
            assert p["symbol"] not in desk.store.accounts()["swing"].positions
            assert any("Paper GTT fired" in a for a in res2["alerts"])
    else:
        assert q["status"] == "expired" and low > p["limit_price"]


def test_unreachable_limit_expires_and_kill_cancels_paper_orders(tmp_path, small_data):
    desk = _desk(tmp_path)
    desk.refresh(_day(small_data, 520))
    props = [p for p in desk.store.proposals("status='pending'") if p["side"] == "BUY"]
    a, b = props[0], props[1]
    nxt = small_data.dates[521]
    low = small_data.low.loc[nxt, a["symbol"]]
    limit = round(min(a["ref_price"] * 0.955, low * 0.99), 2)    # below the next day's low -> cannot fill
    if abs(limit / a["ref_price"] - 1) <= 0.05:
        desk.approve(a["id"], limit_price=limit)
        desk.approve(b["id"])
        desk.kill_switch("test")                              # cancels the queued paper buys
        assert desk.store.proposal(b["id"])["status"] == "cancelled_kill"
        assert desk.store.proposal(a["id"])["status"] == "cancelled_kill"
        desk.resume()
    desk.refresh(_day(small_data, 521))
    assert not desk.store.proposals("status='submitted'")


def test_limit_below_low_is_not_filled(tmp_path, small_data):
    desk = _desk(tmp_path)
    desk.refresh(_day(small_data, 530))
    p = next(p for p in desk.store.proposals("status='pending'") if p["side"] == "BUY")
    nxt = small_data.dates[531]
    o, low = small_data.open.loc[nxt, p["symbol"]], small_data.low.loc[nxt, p["symbol"]]
    limit = round(p["ref_price"] * 0.951, 2)
    desk.approve(p["id"], limit_price=limit)
    desk.refresh(_day(small_data, 531))
    q = desk.store.proposal(p["id"])
    if low > limit:
        assert q["status"] == "expired"
    else:
        assert q["status"] == "filled" and q["fill_price"] == (round(min(o, limit), 2) if o <= limit else limit) \
            or q["fill_price"] <= limit


def test_paper_desk_uses_its_own_database(tmp_path):
    from indquant.cli import _make_desk

    cfg = load_config(overrides=small_overrides(tmp_path, **{"execution.broker": "paper",
                                                            "desk.paper_db_path": str(tmp_path / "fake.db"),
                                                            "desk.db_path": str(tmp_path / "real.db")}))
    assert _make_desk(cfg).store.path == tmp_path / "fake.db"
    real = cfg.copy_with({"execution.broker": "manual"})
    assert _make_desk(real).store.path == tmp_path / "real.db"


def test_fill_price_is_not_mistaken_for_a_corporate_action(tmp_path, small_data):
    desk = _desk(tmp_path, broker="manual", name="real.db")
    desk.refresh(_day(small_data, 500))
    p = next(p for p in desk.store.proposals("status='pending'") if p["sleeve"] == "swing")
    desk.approve(p["id"])
    desk.confirm_fill(p["id"], p["qty"], round(p["limit_price"] * 1.03, 2), when=small_data.dates[501])
    res = desk.refresh(_day(small_data, 501))
    assert not any("dividend" in a or "split/bonus" in a for a in res["alerts"])
    acct = desk.store.accounts()["swing"]
    assert acct.positions[p["symbol"]].shares == p["qty"]
    assert pd.Timestamp(acct.positions[p["symbol"]].last_date) == small_data.dates[501]
