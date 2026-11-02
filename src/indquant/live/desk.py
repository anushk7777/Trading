"""Trade Desk: the human-in-the-loop execution layer for real money.

Daily cycle (after the close):
  refresh  -> the same strategy code as the backtest proposes orders for the next session
              against your REAL book; stop / target breaches become exit proposals;
              BUYs get an AI red-flag review; everything is journaled.
  approve  -> hard pre-trade risk checks, then the broker adapter (manual instructions
              or Zerodha) - nothing ever reaches a broker without your click.
  fill     -> your real fill is booked exactly like a backtest fill (FIFO lots, charges),
              and protective GTT stop-loss / target instructions are produced.
  kill     -> STOP: cancels pending buys, blocks new ones; protective stops stay in place.
  flatten  -> exit everything (requires typing SELL ALL).

Every action lands in an append-only audit log. Proposals, fills and vetoed candidates
are later "settled" against what the market actually did, so you can see whether each
filter and the AI reviewer are earning their keep.
"""
from __future__ import annotations

import copy
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.account import Account, apply_buy, apply_sell
from ..backtest.costs import CostModel
from ..backtest.engine import BarContext, Engine
from ..backtest.runner import build_strategies
from ..execution.brokers import make_broker
from ..execution.market import round_tick
from ..execution.risk_checks import pre_trade_checks

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (sleeve TEXT PRIMARY KEY, json TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, data_date TEXT NOT NULL, sleeve TEXT NOT NULL,
    symbol TEXT NOT NULL, company TEXT, side TEXT NOT NULL, qty INTEGER NOT NULL, ref_price REAL NOT NULL,
    limit_price REAL NOT NULL, stop_dist REAL, target_dist REAL, atr REAL, reason TEXT, score REAL,
    risk_inr REAL, risk_pct REAL, review TEXT, verdict TEXT, status TEXT NOT NULL, broker_ref TEXT,
    message TEXT, fill_qty INTEGER, fill_price REAL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS protections (
    sleeve TEXT NOT NULL, symbol TEXT NOT NULL, ref TEXT, stop REAL, target REAL, qty INTEGER, message TEXT,
    updated TEXT NOT NULL, PRIMARY KEY (sleeve, symbol)
);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, action TEXT NOT NULL,
    detail TEXT);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT, data_date TEXT NOT NULL, sleeve TEXT NOT NULL, symbol TEXT NOT NULL,
    category TEXT NOT NULL, score REAL, settled TEXT, UNIQUE (data_date, sleeve, symbol, category)
);
CREATE TABLE IF NOT EXISTS review_cache (key TEXT PRIMARY KEY, json TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity (date TEXT NOT NULL, sleeve TEXT NOT NULL, equity REAL, cash REAL,
    PRIMARY KEY (date, sleeve));
CREATE TABLE IF NOT EXISTS fills (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, sleeve TEXT, symbol TEXT,
    side TEXT, qty INTEGER, price REAL, costs REAL, pnl REAL, reason TEXT);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DeskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def get(self, key, default=None):
        with self.conn() as c:
            r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def put(self, key, value):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, default=str)))

    def event(self, action: str, **detail):
        with self.conn() as c:
            c.execute("INSERT INTO events (ts, action, detail) VALUES (?,?,?)",
                      (_now(), action, json.dumps(detail, default=str, ensure_ascii=False)))

    def accounts(self) -> dict[str, Account]:
        with self.conn() as c:
            rows = c.execute("SELECT sleeve, json FROM accounts").fetchall()
        return {r["sleeve"]: Account.from_dict(json.loads(r["json"])) for r in rows}

    def save_account(self, acct: Account):
        acct.trades, acct.history = [], []
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO accounts VALUES (?,?,?)",
                      (acct.name, json.dumps(acct.to_dict(), default=str), _now()))

    def proposals(self, where: str = "1=1", params=()) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(f"SELECT * FROM proposals WHERE {where} ORDER BY id", params)]

    def proposal(self, pid: int) -> dict:
        rows = self.proposals("id=?", (pid,))
        if not rows:
            raise KeyError(f"No proposal {pid}")
        return rows[0]

    def add_proposal(self, p: dict) -> int:
        p = {**p, "created": _now(), "updated": _now()}
        cols = ",".join(p)
        with self.conn() as c:
            cur = c.execute(f"INSERT INTO proposals ({cols}) VALUES ({','.join('?' * len(p))})", tuple(p.values()))
            return int(cur.lastrowid)

    def update_proposal(self, pid: int, **fields):
        fields["updated"] = _now()
        sets = ",".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE proposals SET {sets} WHERE id=?", (*fields.values(), pid))

    def protections(self) -> dict[tuple, dict]:
        with self.conn() as c:
            return {(r["sleeve"], r["symbol"]): dict(r) for r in c.execute("SELECT * FROM protections")}

    def set_protection(self, sleeve, symbol, ref, stop, target, qty, message):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO protections VALUES (?,?,?,?,?,?,?,?)",
                      (sleeve, symbol, ref, stop, target, qty, message, _now()))

    def drop_protection(self, sleeve, symbol):
        with self.conn() as c:
            c.execute("DELETE FROM protections WHERE sleeve=? AND symbol=?", (sleeve, symbol))

    def journal_add(self, data_date, sleeve, symbol, category, score=None):
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO journal (data_date, sleeve, symbol, category, score) VALUES (?,?,?,?,?)",
                      (data_date, sleeve, symbol, category, score))

    # review cache interface used by RedFlagReviewer
    def cache_get(self, key):
        with self.conn() as c:
            r = c.execute("SELECT json FROM review_cache WHERE key=?", (key,)).fetchone()
        return json.loads(r["json"]) if r else None

    def cache_put(self, key, value):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO review_cache VALUES (?,?,?)", (key, json.dumps(value), _now()))


class _Cache:
    def __init__(self, store):
        self.get, self.put = store.cache_get, store.cache_put


class TradeDesk:
    def __init__(self, cfg, store: DeskStore | None = None, broker=None, reviewer=None):
        self.cfg = cfg
        self.store = store or DeskStore(cfg.desk.db_path)
        self.broker = broker or make_broker(cfg)
        self.costs = CostModel.from_config(cfg.costs)
        self.reviewer = reviewer
        self.last_strategies: dict = {}

    # ------------------------------------------------------------------ setup
    def init(self, capital: float) -> None:
        if self.store.accounts():
            raise ValueError("Desk already initialised (delete the desk database to start over)")
        for name in build_strategies(self.cfg):
            alloc = capital * float(self.cfg.allocation[name])
            acct = Account(name=name, cash=alloc, peak_equity=alloc)
            acct.meta["initial_equity"] = alloc
            self.store.save_account(acct)
        self.store.put("capital", capital)
        self.store.event("init", capital=capital, broker=self.broker.name)

    @property
    def kill(self) -> bool:
        return bool(self.store.get("kill", False))

    def _company(self, symbol: str) -> str:
        names = self.store.get("company_names", {}) or {}
        return names.get(symbol, symbol)

    # ------------------------------------------------------------------ refresh
    def refresh(self, data, universe: pd.DataFrame | None = None) -> dict:
        cfg = self.cfg
        accounts = self.store.accounts()
        if not accounts:
            raise ValueError("Desk not initialised - run `indquant desk init --capital ...`")
        if universe is not None and "Company" in universe:
            self.store.put("company_names", dict(zip(universe["Symbol"], universe["Company"])))
        i = len(data.dates) - 1
        day = str(data.dates[i].date())
        engine = Engine(data, self.costs)
        strategies = build_strategies(cfg)
        for s in strategies.values():
            s.prepare(data)
        self.last_strategies = strategies
        close = engine.close_ff.iloc[i]
        low, high = data.low.iloc[i], data.high.iloc[i]

        # stale proposals from earlier sessions expire; same-day pending ones are regenerated
        for p in self.store.proposals("status IN ('pending','blocked_ai')"):
            self.store.update_proposal(p["id"], status="expired" if p["data_date"] != day else "replaced")
        open_ = self.store.proposals("status IN ('awaiting_fill','submitted','dry_run')")
        alerts, new_ids = [], []
        total_equity = sum(a.equity(close) for a in accounts.values())

        for name, strat in strategies.items():
            acct = accounts.get(name)
            if acct is None:
                continue
            for sym, pos in acct.positions.items():
                if not np.isnan(close.get(sym, np.nan)):
                    pos.last_price = float(close[sym])
            equity = acct.equity(close)
            acct.peak_equity = max(acct.peak_equity, equity)
            with self.store.conn() as c:
                c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", (day, name, equity, acct.cash))

            work = copy.deepcopy(acct)
            busy = {p["symbol"] for p in open_ if p["sleeve"] == name}
            user_skipped = set()
            if name == "core" and not strat.flags[i]:
                user_skipped = self._maybe_catch_up(work, open_, day)
            work.cash -= sum(p["qty"] * p["limit_price"] for p in open_ if p["sleeve"] == name and p["side"] == "BUY")
            orders = strat.on_bar(i, BarContext(engine, i, work, equity))
            acct.meta = work.meta
            for sym, pos in acct.positions.items():  # e.g. break-even stop moves
                new_stop = work.positions[sym].stop if sym in work.positions else pos.stop
                if new_stop is not None and pos.stop is not None and new_stop > pos.stop + 1e-9:
                    pos.stop = new_stop
                    alerts.append(self._reprotect(name, sym, pos, f"stop raised to ₹{new_stop:,.2f}"))

            exits = set()
            for sym, pos in acct.positions.items():  # protective levels vs today's bar
                if sym in busy:
                    continue
                if pos.stop is not None and float(close.get(sym, np.inf)) <= pos.stop:
                    exits.add(sym)
                    alerts.append(f"{name}/{sym}: closed at/below stop ₹{pos.stop:,.2f}. If your GTT did not fire, "
                                  "exit at the open (proposal created).")
                    new_ids.append(self._propose(day, name, sym, "SELL", pos.shares, float(close[sym]), "stop_breach"))
                elif pos.stop is not None and float(low.get(sym, np.inf)) <= pos.stop:
                    alerts.append(f"{name}/{sym}: traded through stop ₹{pos.stop:,.2f} today - your GTT has probably "
                                  "triggered. Confirm with 'Stopped out' if so.")
                elif pos.target is not None and float(high.get(sym, -np.inf)) >= pos.target:
                    alerts.append(f"{name}/{sym}: reached target ₹{pos.target:,.2f} today - confirm the exit if it filled.")
            other = {s_ for n, a in accounts.items() if n != name for s_ in a.positions} | \
                {p["symbol"] for p in self.store.proposals("status IN ('pending','awaiting_fill','submitted','dry_run') "
                                                           "AND side='BUY' AND sleeve<>?", (name,))}
            for o in orders:
                if o.symbol in busy or o.symbol in exits:
                    continue
                if o.shares > 0 and o.symbol in user_skipped:
                    continue  # catch-up never re-proposes what the user chose to skip
                if o.shares > 0 and name == "swing" and o.symbol in other:
                    self.store.journal_add(day, name, o.symbol, "skip:held_by_other_sleeve")
                    continue
                if o.shares > 0 and self.kill:
                    self.store.journal_add(day, name, o.symbol, "blocked_kill")
                    continue
                new_ids.append(self._propose(day, name, o.symbol, "BUY" if o.shares > 0 else "SELL", abs(o.shares),
                                             float(o.ref_price or close[o.symbol]), o.reason, o.stop_dist,
                                             o.target_dist, o.atr, total_equity))
            for d in getattr(strat, "last_decisions", []) or []:
                if d.get("decision") == "rejected" and d.get("symbol") not in (None, "*"):
                    self.store.journal_add(day, name, d["symbol"], f"veto:{d['reason']}", d.get("score"))
            self.store.save_account(acct)

        reviewed = self._review(day, new_ids, data)
        self.store.put("last_refresh", {"data_date": day, "at": _now(), "alerts": alerts})
        self.store.event("refresh", data_date=day, proposals=len(new_ids), reviewed=reviewed)
        return {"data_date": day, "proposals": len(new_ids), "alerts": alerts, "reviewed": reviewed}

    def _maybe_catch_up(self, work: Account, open_: list[dict], day: str) -> set:
        """Re-propose core targets that were never filled (skipped by kill switch, expired, cancelled),
        but never ones the user deliberately skipped since the last rebalance."""
        targets = work.meta.get("last_targets") or {}
        last = work.meta.get("last_rebalance")
        if not targets or not last or last == day:
            return set()
        with self.store.conn() as c:
            skipped = {r["symbol"] for r in c.execute(
                "SELECT symbol FROM journal WHERE sleeve='core' AND category='skipped_by_user' AND data_date>=?", (last,))}
        pending = {p["symbol"] for p in open_ if p["sleeve"] == "core"}
        missing = [s_ for s_, w in targets.items() if w > 0 and s_ not in work.positions and s_ not in pending
                   and s_ not in skipped]
        if missing:
            work.meta["force_rebalance"] = True
            self.store.event("core_catch_up", missing=missing)
        return skipped

    def _propose(self, day, sleeve, symbol, side, qty, ref, reason, stop_dist=None, target_dist=None, atr=None,
                 total_equity=None) -> int:
        ex = self.cfg.execution
        buf = float(ex.limit_buffer)
        limit = round_tick(ref * (1 + buf), float(ex.tick_size), "up") if side == "BUY" else \
            round_tick(ref * (1 - buf), float(ex.tick_size), "down")
        risk = qty * stop_dist if (side == "BUY" and stop_dist) else None
        capital = total_equity or float(self.store.get("capital", 1.0))
        pid = self.store.add_proposal(dict(
            data_date=day, sleeve=sleeve, symbol=symbol, company=self._company(symbol), side=side, qty=int(qty),
            ref_price=ref, limit_price=limit, stop_dist=stop_dist, target_dist=target_dist, atr=atr, reason=reason,
            risk_inr=risk, risk_pct=(risk / capital) if risk else None, status="pending"))
        if side == "BUY":
            self.store.journal_add(day, sleeve, symbol, "proposed")
        return pid

    def _review(self, day, ids, data) -> int:
        if self.reviewer is None or not self.cfg.reviewer.enabled:
            return 0
        props = [p for p in (self.store.proposal(i) for i in ids) if p["side"] == "BUY"]
        props.sort(key=lambda p: -p["qty"] * p["limit_price"])
        done = 0
        usage = self.store.get("ai_usage", {}) or {}
        for p in props[: int(self.cfg.reviewer.max_reviews_per_run)]:
            cand = self._candidate(p, data)
            v = self.reviewer.review(cand, day)
            tin, tout = usage.get(day, [0, 0])
            usage[day] = [tin + v.input_tokens, tout + v.output_tokens]
            self.store.put("ai_usage", usage)
            status = "blocked_ai" if v.verdict == "VETO" else "pending"
            self.store.update_proposal(p["id"], verdict=v.verdict, review=json.dumps(v.to_dict()), status=status)
            if v.verdict == "VETO":
                self.store.journal_add(day, p["sleeve"], p["symbol"], "ai_veto")
            done += 1
        return done

    def _candidate(self, p, data) -> dict:
        c = data.close[p["symbol"]].dropna()
        hi = c.iloc[-252:].max()
        accts = self.store.accounts()
        held = accts[p["sleeve"]].positions.get(p["symbol"]) if p["sleeve"] in accts else None
        total = sum(a.equity(data.close.ffill().iloc[-1]) for a in accts.values())
        return {
            "symbol": p["symbol"], "company": p["company"], "sector": data.sectors.get(p["symbol"], "Unknown"),
            "sleeve": p["sleeve"], "proposed_action": f"BUY {p['qty']} @ limit {p['limit_price']}",
            "last_close": round(float(c.iloc[-1]), 2),
            "return_1m": f"{c.iloc[-1] / c.iloc[-22] - 1:.1%}" if len(c) > 22 else None,
            "return_3m": f"{c.iloc[-1] / c.iloc[-64] - 1:.1%}" if len(c) > 64 else None,
            "pct_from_52w_high": f"{c.iloc[-1] / hi - 1:.1%}",
            "atr_pct": f"{p['atr'] / c.iloc[-1]:.1%}" if p.get("atr") else None,
            "proposed_stop": round(p["ref_price"] - p["stop_dist"], 2) if p.get("stop_dist") else None,
            "proposed_target": round(p["ref_price"] + p["target_dist"], 2) if p.get("target_dist") else None,
            "position_value_pct": f"{p['qty'] * p['limit_price'] / total:.1%}" if total else None,
            "held_now": f"{held.shares} shares" if held else "none",
        }

    # ------------------------------------------------------------------ actions
    def approve(self, pid: int, override_reason: str | None = None, qty: int | None = None,
                limit_price: float | None = None) -> dict:
        p = self.store.proposal(pid)
        if p["status"] == "blocked_ai" and not override_reason:
            raise ValueError("AI reviewer vetoed this trade - approving needs an override reason")
        if p["status"] not in ("pending", "blocked_ai"):
            raise ValueError(f"Proposal {pid} is {p['status']}, not pending")
        if qty is not None or limit_price is not None:
            p["qty"] = int(qty if qty is not None else p["qty"])
            p["limit_price"] = float(limit_price if limit_price is not None else p["limit_price"])
            self.store.update_proposal(pid, qty=p["qty"], limit_price=p["limit_price"])
        accts = self.store.accounts()
        acct = accts[p["sleeve"]]
        prices = pd.Series({s: pos.last_price or pos.entry_price for a in accts.values() for s, pos in a.positions.items()})
        total = sum(a.equity(prices) for a in accts.values())
        today = str(datetime.now().date())
        sent_today = len(self.store.proposals("status IN ('awaiting_fill','submitted','dry_run','filled') "
                                              "AND substr(updated,1,10)=?", (today,)))
        problems = pre_trade_checks(p, kill=self.kill, total_equity=total,
                                    held_qty=acct.positions[p["symbol"]].shares if p["symbol"] in acct.positions else 0,
                                    sleeve_cash=acct.cash, orders_today=sent_today, ex=self.cfg.execution)
        if problems:
            self.store.update_proposal(pid, status="rejected_risk", message="; ".join(problems))
            self.store.event("risk_reject", id=pid, problems=problems)
            return {"status": "rejected_risk", "message": "; ".join(problems)}
        res = self.broker.submit(p)
        self.store.update_proposal(pid, status=res["status"], broker_ref=res.get("ref"), message=res.get("message"))
        self.store.event("approve", id=pid, symbol=p["symbol"], side=p["side"], qty=p["qty"],
                         limit=p["limit_price"], override=override_reason, broker=self.broker.name, result=res)
        if override_reason:
            self.store.journal_add(p["data_date"], p["sleeve"], p["symbol"], "ai_override")
        return res

    def skip(self, pid: int, reason: str = "") -> None:
        p = self.store.proposal(pid)
        if p["status"] not in ("pending", "blocked_ai"):
            raise ValueError(f"Proposal {pid} is {p['status']}")
        self.store.update_proposal(pid, status="skipped", message=reason)
        self.store.journal_add(p["data_date"], p["sleeve"], p["symbol"], "skipped_by_user")
        self.store.event("skip", id=pid, reason=reason)

    def cancel(self, pid: int) -> None:
        p = self.store.proposal(pid)
        if p["status"] not in ("awaiting_fill", "submitted", "dry_run"):
            raise ValueError(f"Proposal {pid} is {p['status']}")
        self.store.update_proposal(pid, status="cancelled")
        self.store.event("cancel", id=pid)

    def confirm_fill(self, pid: int, qty: int, price: float, when=None) -> dict:
        p = self.store.proposal(pid)
        if p["status"] not in ("awaiting_fill", "submitted", "dry_run"):
            raise ValueError(f"Proposal {pid} is {p['status']}, cannot fill")
        accts = self.store.accounts()
        acct = accts[p["sleeve"]]
        when = when or datetime.now()
        cost = self.costs.cost("buy" if p["side"] == "BUY" else "sell", qty * price)
        message = ""
        if p["side"] == "BUY":
            existing = acct.positions.get(p["symbol"])
            apply_buy(acct, p["symbol"], int(qty), float(price), cost, when, p["stop_dist"], p["target_dist"],
                      p["atr"], p["reason"])
            pos = acct.positions[p["symbol"]]
            if existing is None or pos.stop is not None:
                message = self._reprotect(p["sleeve"], p["symbol"], pos, "new position")
            trade = acct.trades[-1]
        else:
            trade = apply_sell(acct, p["symbol"], int(qty), float(price), cost, when, p["reason"])
            message = self._after_exit(p["sleeve"], p["symbol"], acct)
        self.store.save_account(acct)
        self.store.update_proposal(pid, status="filled", fill_qty=int(qty), fill_price=float(price), message=message)
        self._record_fill(p["sleeve"], trade)
        self.store.event("fill", id=pid, symbol=p["symbol"], side=p["side"], qty=qty, price=price, cost=cost)
        return {"status": "filled", "message": message}

    def stopped_out(self, sleeve: str, symbol: str, price: float, reason: str = "stop") -> dict:
        """A protective GTT executed at the broker - book it."""
        accts = self.store.accounts()
        acct = accts[sleeve]
        pos = acct.positions.get(symbol)
        if pos is None:
            raise ValueError(f"No {symbol} position in {sleeve}")
        trade = apply_sell(acct, symbol, pos.shares, float(price), self.costs.cost("sell", pos.shares * price),
                           datetime.now(), reason)
        message = self._after_exit(sleeve, symbol, acct)
        self.store.save_account(acct)
        self._record_fill(sleeve, trade)
        for p in self.store.proposals("sleeve=? AND symbol=? AND side='SELL' AND status IN ('pending')", (sleeve, symbol)):
            self.store.update_proposal(p["id"], status="superseded", message="position closed by protective order")
        self.store.event("stopped_out", sleeve=sleeve, symbol=symbol, price=price, reason=reason)
        return {"status": "closed", "message": message}

    def exit_position(self, sleeve: str, symbol: str, reason: str = "manual_exit") -> dict:
        acct = self.store.accounts()[sleeve]
        pos = acct.positions.get(symbol)
        if pos is None:
            raise ValueError(f"No {symbol} position in {sleeve}")
        pid = self._propose(str(datetime.now().date()), sleeve, symbol, "SELL", pos.shares,
                            float(pos.last_price or pos.entry_price), reason)
        res = self.approve(pid)
        return {"id": pid, **res}

    def set_stop(self, sleeve: str, symbol: str, stop: float | None, target: float | None = None) -> dict:
        acct = self.store.accounts()[sleeve]
        pos = acct.positions.get(symbol)
        if pos is None:
            raise ValueError(f"No {symbol} position in {sleeve}")
        ref = pos.last_price or pos.entry_price
        if stop is not None and stop >= ref:
            raise ValueError("A long position's stop must be below the last price")
        pos.stop = stop
        if target is not None:
            pos.target = target
        message = self._reprotect(sleeve, symbol, pos, "manual stop change")
        self.store.save_account(acct)
        self.store.event("set_stop", sleeve=sleeve, symbol=symbol, stop=stop, target=target)
        return {"message": message}

    def kill_switch(self, reason: str = "") -> dict:
        self.store.put("kill", True)
        n = 0
        for p in self.store.proposals("side='BUY' AND status IN ('pending','blocked_ai')"):
            self.store.update_proposal(p["id"], status="cancelled_kill")
            n += 1
        res = self.broker.cancel_open_orders()
        self.store.event("KILL", reason=reason, cancelled_proposals=n, broker=res)
        return {"message": f"STOPPED. {n} pending buy(s) cancelled. {res['message']} "
                           "Protective stop-losses stay active. Use RESUME to trade again."}

    def resume(self) -> dict:
        self.store.put("kill", False)
        self.store.event("resume")
        return {"message": "Trading resumed."}

    def flatten(self, confirm: str) -> dict:
        if confirm != "SELL ALL":
            raise ValueError("Type SELL ALL to confirm flattening every position")
        self.kill_switch("flatten")
        results = []
        for sleeve, acct in self.store.accounts().items():
            for sym in list(acct.positions):
                results.append({"symbol": sym, **self.exit_position(sleeve, sym, "flatten")})
        return {"message": f"{len(results)} exit order(s) created.", "results": results}

    def force_rebalance(self) -> None:
        accts = self.store.accounts()
        if "core" in accts:
            accts["core"].meta["force_rebalance"] = True
            self.store.save_account(accts["core"])
            self.store.event("force_rebalance")

    def sync(self) -> dict:
        """Pull fills / triggered GTTs from an API broker and book them."""
        booked = []
        refs = {p["broker_ref"]: p for p in self.store.proposals("status='submitted'")}
        for f in self.broker.fills():
            p = refs.get(f["ref"])
            if p:
                self.confirm_fill(p["id"], f["qty"], f["price"])
                booked.append(p["symbol"])
        triggered = set(self.broker.triggered_protections())
        for (sleeve, sym), prot in self.store.protections().items():
            if prot["ref"] in triggered:
                acct = self.store.accounts()[sleeve]
                pos = acct.positions.get(sym)
                if pos:
                    price = prot["stop"] if (pos.last_price or 0) <= (prot["stop"] or 0) * 1.02 else prot["target"]
                    self.stopped_out(sleeve, sym, float(price or pos.last_price), "gtt_triggered")
                    booked.append(sym)
        self.store.event("sync", booked=booked)
        return {"booked": booked}

    # ------------------------------------------------------------------ helpers
    def _reprotect(self, sleeve, symbol, pos, why) -> str:
        old = self.store.protections().get((sleeve, symbol))
        if old and old.get("ref"):
            self.broker.remove_protection(old["ref"], symbol)
        res = self.broker.protect(symbol, pos.shares, pos.stop, pos.target, pos.last_price or pos.entry_price)
        if res.get("message"):
            self.store.set_protection(sleeve, symbol, res.get("ref"), pos.stop, pos.target, pos.shares, res["message"])
        msg = f"{sleeve}/{symbol} ({why}): {res.get('message', '')}"
        if old and self.broker.name == "manual":
            msg += " Delete the old GTT for this stock first."
        return msg

    def _after_exit(self, sleeve, symbol, acct) -> str:
        if symbol in acct.positions:
            return self._reprotect(sleeve, symbol, acct.positions[symbol], "partial exit")
        prot = self.store.protections().get((sleeve, symbol))
        self.store.drop_protection(sleeve, symbol)
        if prot:
            return self.broker.remove_protection(prot.get("ref"), symbol)["message"]
        return ""

    def _record_fill(self, sleeve, trade):
        if not trade:
            return
        with self.store.conn() as c:
            c.execute("INSERT INTO fills (ts, sleeve, symbol, side, qty, price, costs, pnl, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                      (_now(), sleeve, trade["symbol"], trade["side"], trade["shares"], trade["price"], trade["costs"],
                       trade["pnl"], trade["reason"]))

    # ------------------------------------------------------------------ views
    def state(self) -> dict:
        accts = self.store.accounts()
        last = self.store.get("last_refresh", {}) or {}
        protections = self.store.protections()
        positions = []
        for sleeve, a in accts.items():
            for sym, pos in a.positions.items():
                lp = pos.last_price or pos.entry_price
                prot = protections.get((sleeve, sym), {})
                positions.append({
                    "sleeve": sleeve, "symbol": sym, "shares": pos.shares, "entry_date": pos.entry_date,
                    "avg_cost": round(pos.avg_cost, 2), "last": round(lp, 2),
                    "pnl": round((lp - pos.avg_cost) * pos.shares, 0),
                    "pnl_pct": round(lp / pos.avg_cost - 1, 4) if pos.avg_cost else None,
                    "stop": round(pos.stop, 2) if pos.stop else None, "target": round(pos.target, 2) if pos.target else None,
                    "risk_to_stop": round(max(lp - pos.stop, 0) * pos.shares, 0) if pos.stop else None,
                    "protection": prot.get("message", "")})
        prices = pd.Series({p["symbol"]: p["last"] for p in positions}, dtype=float)
        sleeves = {n: {"equity": round(a.equity(prices), 0), "cash": round(a.cash, 0),
                       "halted": bool(a.meta.get("halted")), "loss_streak": a.meta.get("loss_streak", 0),
                       "last_rebalance": a.meta.get("last_rebalance")} for n, a in accts.items()}
        props = self.store.proposals("status IN ('pending','blocked_ai','awaiting_fill','submitted','dry_run',"
                                     "'rejected_risk') AND data_date >= ?", (last.get("data_date", "0000"),))
        for p in props:
            p["review"] = json.loads(p["review"]) if p.get("review") else None
        with self.store.conn() as c:
            events = [dict(r) for r in c.execute("SELECT * FROM events ORDER BY id DESC LIMIT 25")]
        rc = self.cfg.reviewer
        usage = self.store.get("ai_usage", {}) or {}
        price = lambda t: t[0] / 1e6 * float(rc.price_in_per_mtok) + t[1] / 1e6 * float(rc.price_out_per_mtok)  # noqa: E731
        cost = {"last_day_usd": round(price(usage.get(last.get("data_date"), [0, 0])), 4),
                "total_usd": round(sum(price(t) for t in usage.values()), 4)}
        return {
            "broker": self.broker.name, "dry_run": bool(getattr(self.broker, "dry_run", False)),
            "kill": self.kill, "capital": self.store.get("capital"), "data_date": last.get("data_date"),
            "refreshed_at": last.get("at"), "alerts": last.get("alerts", []),
            "equity": round(sum(s["equity"] for s in sleeves.values()), 0), "sleeves": sleeves,
            "proposals": props, "positions": positions, "events": events, "ai_cost_usd": cost,
            "reviewer": {"enabled": bool(self.cfg.reviewer.enabled), "model": self.cfg.reviewer.model},
        }

    # ------------------------------------------------------------------ journal
    def settle_journal(self, data) -> pd.DataFrame:
        """Score every journaled decision against what the market did next (close-to-close vs benchmark)."""
        horizons = [int(h) for h in self.cfg.desk.journal_horizons]
        dates = data.dates
        with self.store.conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM journal")]
        out = []
        for r in rows:
            settled = json.loads(r["settled"]) if r["settled"] else {}
            k = int(dates.searchsorted(pd.Timestamp(r["data_date"])))
            if r["symbol"] in data.close.columns and k < len(dates):
                c0 = data.close[r["symbol"]].iloc[k]
                b0 = data.benchmark.iloc[k]
                for h in horizons:
                    if str(h) in settled or k + h >= len(dates):
                        continue
                    c1, b1 = data.close[r["symbol"]].iloc[k + h], data.benchmark.iloc[k + h]
                    if np.isnan(c0) or np.isnan(c1):
                        continue
                    ret = float(c1 / c0 - 1)
                    settled[str(h)] = {"ret": ret, "alpha": ret - float(b1 / b0 - 1)}
                with self.store.conn() as c:
                    c.execute("UPDATE journal SET settled=? WHERE id=?", (json.dumps(settled), r["id"]))
            for h, v in settled.items():
                out.append({"category": r["category"], "horizon": int(h), "alpha": v["alpha"], "ret": v["ret"]})
        df = pd.DataFrame(out)
        if df.empty:
            return df
        g = df.groupby(["category", "horizon"])
        summary = g["alpha"].agg(["count", "mean"]).rename(columns={"mean": "mean_alpha"})
        summary["hit_rate"] = g["alpha"].apply(lambda s: (s > 0).mean())
        summary["mean_return"] = g["ret"].mean()
        return summary.reset_index()
