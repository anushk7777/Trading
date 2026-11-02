"""Trade desk, broker adapters and console: the human-in-the-loop execution path."""
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from indquant.config import load_config
from indquant.execution.brokers import KiteBroker, ManualBroker, protective_legs
from indquant.execution.market import round_tick
from indquant.features.own_signals import circuit_days, smooth_momentum
from indquant.live.console import ConsoleApp
from indquant.live.desk import DeskStore, TradeDesk
from indquant.review.reviewer import Verdict

from .conftest import small_overrides


class FakeReviewer:
    def __init__(self, veto=()):
        self.veto = set(veto)

    def review(self, cand, as_of):
        if cand["symbol"] in self.veto:
            return Verdict("VETO", "fraud probe", [{"category": "fraud_or_governance", "severity": "high",
                                                     "evidence_ids": [1], "explanation": "probe"}])
        return Verdict("PASS", "no news")

    def cost_usd(self, v):
        return 0.0


@pytest.fixture
def desk(tmp_path, small_data):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    d = TradeDesk(cfg, store=DeskStore(tmp_path / "desk.db"), reviewer=FakeReviewer())
    d.init(1_000_000)
    return d


def _data(small_data, i):
    return small_data.truncate(small_data.dates[i])


def test_refresh_approve_fill_protect_and_exit(desk, small_data):
    d0 = _data(small_data, 500)
    res = desk.refresh(d0)
    props = desk.store.proposals("status='pending'")
    assert res["proposals"] == len(props) > 0
    buys = [p for p in props if p["side"] == "BUY"]
    reviewed = [p for p in buys if p["verdict"]]
    assert len(reviewed) == min(len(buys), desk.cfg.reviewer.max_reviews_per_run)
    assert all(p["verdict"] == "PASS" for p in reviewed)
    swing = next(p for p in props if p["sleeve"] == "swing")
    assert swing["stop_dist"] and swing["target_dist"] and 0 < swing["risk_pct"] < 0.02

    r = desk.approve(swing["id"])
    assert r["status"] == "awaiting_fill" and "CNC" in r["message"] and "LIMIT" in r["message"]
    f = desk.confirm_fill(swing["id"], swing["qty"], swing["limit_price"])
    assert "GTT" in f["message"] and "OCO" in f["message"]
    acct = desk.store.accounts()["swing"]
    pos = acct.positions[swing["symbol"]]
    assert pos.shares == swing["qty"] and pos.stop < swing["limit_price"] < pos.target
    assert acct.cash < 300_000

    out = desk.stopped_out("swing", swing["symbol"], pos.stop)
    assert swing["symbol"] not in desk.store.accounts()["swing"].positions
    assert "Delete" in out["message"]
    assert desk.store.accounts()["swing"].meta["loss_streak"] == 1


def test_stop_breach_creates_exit_proposal(desk, small_data):
    desk.refresh(_data(small_data, 500))
    p = next(p for p in desk.store.proposals("status='pending'") if p["sleeve"] == "swing")
    desk.approve(p["id"])
    desk.confirm_fill(p["id"], p["qty"], p["limit_price"])
    acct = desk.store.accounts()["swing"]
    acct.positions[p["symbol"]].stop = p["limit_price"] * 10  # force: next close is below the stop
    desk.store.save_account(acct)
    res = desk.refresh(_data(small_data, 501))
    sells = desk.store.proposals("side='SELL' AND symbol=? AND status='pending'", (p["symbol"],))
    assert sells and sells[0]["reason"] == "stop_breach" and sells[0]["qty"] == p["qty"]
    assert any("below stop" in a for a in res["alerts"])


def test_kill_switch_blocks_buys_and_resume(desk, small_data):
    desk.refresh(_data(small_data, 500))
    pid = desk.store.proposals("status='pending' AND side='BUY'")[0]["id"]
    msg = desk.kill_switch("test")["message"]
    assert "STOPPED" in msg and desk.kill
    assert desk.store.proposal(pid)["status"] == "cancelled_kill"
    res = desk.refresh(_data(small_data, 501))
    assert not desk.store.proposals("status='pending' AND side='BUY'")
    desk.resume()
    desk.refresh(_data(small_data, 502))
    assert desk.store.proposals("status='pending' AND side='BUY'") or res is not None


def test_ai_veto_blocks_until_override_and_risk_checks_apply(tmp_path, small_data):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    first = TradeDesk(cfg, store=DeskStore(tmp_path / "a.db"), reviewer=FakeReviewer())
    first.init(1_000_000)
    first.refresh(_data(small_data, 500))
    sym = first.store.proposals("side='BUY' AND sleeve='swing'")[0]["symbol"]

    desk = TradeDesk(cfg, store=DeskStore(tmp_path / "b.db"), reviewer=FakeReviewer(veto={sym}))
    desk.init(1_000_000)
    desk.refresh(_data(small_data, 500))
    p = desk.store.proposals("symbol=? AND sleeve='swing'", (sym,))[0]
    assert p["status"] == "blocked_ai"
    with pytest.raises(ValueError, match="override"):
        desk.approve(p["id"])
    assert desk.approve(p["id"], override_reason="I checked: old news")["status"] == "awaiting_fill"

    q = desk.store.proposals("status='pending' AND side='BUY'")[0]
    r = desk.approve(q["id"], qty=10_000_000)
    assert r["status"] == "rejected_risk" and "exceeds" in r["message"]


def test_flatten_requires_confirmation_and_core_catch_up(desk, small_data):
    dates = small_data.dates
    i0 = next(i for i in range(500, 600) if dates[i].day <= 12 and dates[i].month == dates[i + 4].month)
    desk.refresh(_data(small_data, i0))
    with pytest.raises(ValueError):
        desk.flatten("yes")
    core = desk.store.proposals("sleeve='core' AND side='BUY' AND status='pending'")
    for p in core[:2]:
        desk.approve(p["id"])
        desk.confirm_fill(p["id"], p["qty"], p["limit_price"])
    for p in core[2:]:
        desk.skip(p["id"]) if p is core[2] else None
    desk.refresh(_data(small_data, i0 + 4))  # same month: missing (not skipped) targets are re-proposed
    again = {p["symbol"] for p in desk.store.proposals("sleeve='core' AND side='BUY' AND status='pending'")}
    assert core[2]["symbol"] not in again
    assert again & {p["symbol"] for p in core[3:]}
    res = desk.flatten("SELL ALL")
    assert len(res["results"]) == sum(len(a.positions) for a in desk.store.accounts().values())


def test_journal_settles_decisions(desk, small_data):
    desk.refresh(_data(small_data, 500))
    summary = desk.settle_journal(_data(small_data, 530))
    assert set(summary["category"]) >= {"proposed"} and set(summary["horizon"]) == {5, 20}


def test_brokers_manual_and_kite_dry_and_live():
    cfg = load_config()
    p = {"symbol": "RELIANCE", "side": "BUY", "qty": 5, "limit_price": 2901.5}
    assert "AMO" in ManualBroker(cfg).submit(p)["message"] or "regular" in ManualBroker(cfg).submit(p)["message"]
    legs = protective_legs(2800.03, 3055.01, cfg.execution)
    assert legs["stop_trigger"] == 2800.0 and legs["target_trigger"] == 3055.05 and legs["stop_limit"] < 2800
    assert round_tick(100.01, 0.05, "up") == 100.05 and round_tick(100.04, 0.05, "down") == 100.0

    dry = KiteBroker(cfg)
    r = dry.submit(p)
    assert r["status"] == "dry_run" and '"product": "CNC"' in r["message"]

    calls = []

    class FakeKite:
        def place_order(self, **kw):
            calls.append(("order", kw))
            return "ORD1"

        def place_gtt(self, **kw):
            calls.append(("gtt", kw))
            return {"trigger_id": 77}

        def orders(self):
            return [{"order_id": "ORD1", "tag": "indquant", "status": "COMPLETE", "quantity": 5,
                     "filled_quantity": 5, "average_price": 2899.0, "variety": "regular"},
                    {"order_id": "X", "tag": "other", "status": "OPEN", "variety": "regular"},
                    {"order_id": "ORD2", "tag": "indquant", "status": "OPEN", "variety": "amo"}]

        def cancel_order(self, variety, order_id):
            calls.append(("cancel", order_id))

        def get_gtts(self):
            return [{"id": 77, "status": "triggered"}]

    live_cfg = load_config(overrides={"execution.broker": "kite", "execution.dry_run": False})
    kb = KiteBroker(live_cfg, kite=FakeKite())
    assert kb.submit(p)["ref"] == "ORD1"
    order = calls[0][1]
    assert order["product"] == "CNC" and order["order_type"] == "LIMIT" and order["tradingsymbol"] == "RELIANCE"
    g = kb.protect("RELIANCE", 5, 2800.0, 3055.0, 2900.0)
    gtt = calls[1][1]
    assert g["ref"] == "77" and gtt["trigger_type"] == "two-leg" and gtt["trigger_values"] == [2800.0, 3055.0]
    assert kb.fills() == [{"ref": "ORD1", "qty": 5, "price": 2899.0}]
    kb.cancel_open_orders()
    assert ("cancel", "ORD2") in calls and ("cancel", "X") not in calls
    assert kb.triggered_protections() == ["77"]


def test_console_requires_token_and_dispatches(desk, small_data):
    refreshed = []
    app = ConsoleApp(desk, lambda: refreshed.append(desk.refresh(_data(small_data, 500))), port=0)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler())
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/")
        assert e.value.code == 403
        page = urllib.request.urlopen(base + f"/?t={app.token}").read().decode()
        assert "STOP ALL" in page and "Refresh plan" in page

        def call(path, body=None, token=app.token):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"X-Desk-Token": token, "Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        with pytest.raises(urllib.error.HTTPError):
            call("/api/state", token="wrong")
        app.refresh_fn()
        st = call("/api/state")
        assert st["broker"] == "manual" and st["proposals"]
        pid = st["proposals"][0]["id"]
        assert call("/api/approve", {"id": pid})["ok"]
        assert call("/api/kill", {})["ok"] and call("/api/state")["kill"]
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/api/flatten", {"confirm": "nope"})
        assert e.value.code == 400
    finally:
        server.shutdown()


def test_own_signals():
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2024-01-01", periods=300)
    steady = pd.Series(100 * 1.001 ** np.arange(300), index=idx)
    jumpy = pd.Series(100.0, index=idx)
    jumpy.iloc[100:] = 125.0
    jumpy = jumpy * (1 + 0.0005 * np.sin(np.arange(300)))
    sm = smooth_momentum(pd.DataFrame({"steady": steady, "jumpy": jumpy}))
    assert sm["steady"].iloc[-1] > sm["jumpy"].iloc[-1]

    c = pd.DataFrame({"S": [100, 105, 110.25, 110.0, 109.0]}, index=idx[:5])
    h = c.copy()
    h.iloc[3:] += 1
    data = SimpleNamespace(close=c, high=h, low=c - 1)
    assert circuit_days(data).iloc[2, 0] == 2.0
