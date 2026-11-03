"""Broker adapters behind one interface.

* ManualBroker (default, free): nothing is sent anywhere. Each approved order comes back
  with exact click-by-click instructions for your broker app (CNC limit order, then a GTT
  OCO stop-loss / target), and you confirm the fill price in the console.
* KiteBroker (optional): Zerodha Kite Connect "Personal" API (free for order placement;
  SEBI's 2026 retail-algo rules require a registered app and a whitelisted static IP).
  Ships in dry-run mode; every order still needs your click in the console.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date
from pathlib import Path

from .market import market_open, now_ist, round_tick

log = logging.getLogger(__name__)
OPEN_STATUSES = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED", "OPEN PENDING", "VALIDATION PENDING",
                 "PUT ORDER REQ RECEIVED", "MODIFY PENDING"}


def protective_legs(stop: float | None, target: float | None, ex) -> dict:
    tick = float(ex.tick_size)
    out = {}
    if stop:
        out["stop_trigger"] = round_tick(stop, tick, "down")
        out["stop_limit"] = round_tick(stop * (1 - float(ex.stop_limit_buffer)), tick, "down")
    if target:
        out["target_trigger"] = round_tick(target, tick, "up")
        out["target_limit"] = out["target_trigger"]
    return out


class Broker:
    name = "base"
    sends_orders = False
    simulated = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.ex = cfg.execution

    def submit(self, p: dict) -> dict:
        raise NotImplementedError

    def protect(self, symbol: str, qty: int, stop: float | None, target: float | None, last: float) -> dict:
        raise NotImplementedError

    def remove_protection(self, ref: str | None, symbol: str) -> dict:
        return {"message": f"Delete any GTT for {symbol} in your broker app."}

    def cancel_open_orders(self) -> dict:
        return {"message": "Cancel any unfilled buy orders in your broker app."}

    def fills(self) -> list[dict]:
        return []

    def triggered_protections(self) -> list[str]:
        return []


class ManualBroker(Broker):
    name = "manual"

    def submit(self, p: dict) -> dict:
        when = "a regular order" if market_open(holidays=self.ex.holidays) else \
            "an AMO (after-market order) so it goes to the exchange at the next open"
        side = "Buy" if p["side"] == "BUY" else "Sell"
        text = (f"In your broker app: {side} {p['symbol']} (NSE) · Product CNC (delivery) · Order type LIMIT · "
                f"Price ₹{p['limit_price']:,.2f} · Qty {p['qty']}. Place it as {when}. "
                "When it executes, press 'Mark filled' and enter your actual average price.")
        return {"status": "awaiting_fill", "ref": f"MAN-{uuid.uuid4().hex[:8]}", "message": text}

    def protect(self, symbol, qty, stop, target, last) -> dict:
        legs = protective_legs(stop, target, self.ex)
        if not legs:
            return {"ref": None, "message": ""}
        if "target_trigger" in legs:
            text = (f"Create a GTT → OCO for {symbol} (CNC, qty {qty}): stop-loss trigger ₹{legs['stop_trigger']:,.2f} "
                    f"with limit ₹{legs['stop_limit']:,.2f}; target trigger ₹{legs['target_trigger']:,.2f} with limit "
                    f"₹{legs['target_limit']:,.2f}.")
        else:
            text = (f"Create a GTT → Single for {symbol} (CNC, sell qty {qty}): trigger ₹{legs['stop_trigger']:,.2f}, "
                    f"limit ₹{legs['stop_limit']:,.2f} (protective stop).")
        return {"ref": f"MANGTT-{uuid.uuid4().hex[:8]}", "message": text, **legs}


class PaperBroker(Broker):
    """Fake money: approved orders fill automatically against real next-session prices."""
    name = "paper"
    simulated = True

    def submit(self, p: dict) -> dict:
        return {"status": "submitted", "ref": f"PAPER-{uuid.uuid4().hex[:8]}",
                "message": "Paper order queued - it fills automatically at the next session's open if the price "
                           "allows. Refresh after that session to see the result."}

    def protect(self, symbol, qty, stop, target, last) -> dict:
        legs = protective_legs(stop, target, self.ex)
        if not legs:
            return {"ref": None, "message": ""}
        parts = [f"stop ₹{legs['stop_trigger']:,.2f}"] + ([f"target ₹{legs['target_trigger']:,.2f}"]
                                                         if "target_trigger" in legs else [])
        return {"ref": f"PAPERGTT-{uuid.uuid4().hex[:8]}",
                "message": f"Paper GTT armed for {symbol} ({' / '.join(parts)}) - checked automatically.", **legs}

    def remove_protection(self, ref, symbol) -> dict:
        return {"message": f"Paper GTT for {symbol} removed."}

    def cancel_open_orders(self) -> dict:
        return {"message": "Open paper orders cancelled."}


class KiteBroker(Broker):
    name = "kite"
    sends_orders = True

    def __init__(self, cfg, kite=None):
        super().__init__(cfg)
        self.dry_run = bool(self.ex.dry_run)
        self._kite = kite

    # ---------------------------------------------------------------- session
    @property
    def kite(self):
        if self._kite is None:
            from kiteconnect import KiteConnect

            key = os.environ.get(self.ex.kite_api_key_env)
            if not key:
                raise RuntimeError(f"Set {self.ex.kite_api_key_env} (Kite Connect app API key)")
            kite = KiteConnect(api_key=key)
            token = load_kite_token(self.ex.kite_token_file)
            if not token:
                raise RuntimeError("No Kite session for today - run `indquant broker-login`")
            kite.set_access_token(token)
            self._kite = kite
        return self._kite

    def _dry(self, what: str, payload: dict) -> dict:
        log.info("[DRY RUN] %s %s", what, json.dumps(payload, default=str))
        return {"status": "dry_run", "ref": f"DRY-{uuid.uuid4().hex[:8]}",
                "message": f"DRY RUN - would {what}: {json.dumps(payload, default=str)}"}

    # ---------------------------------------------------------------- orders
    def submit(self, p: dict) -> dict:
        k = _KiteConstants
        payload = dict(
            variety=k.VARIETY_REGULAR if market_open(holidays=self.ex.holidays) else k.VARIETY_AMO,
            exchange=k.EXCHANGE_NSE, tradingsymbol=p["symbol"],
            transaction_type=k.TRANSACTION_TYPE_BUY if p["side"] == "BUY" else k.TRANSACTION_TYPE_SELL,
            quantity=int(p["qty"]), product=k.PRODUCT_CNC, order_type=k.ORDER_TYPE_LIMIT,
            price=float(p["limit_price"]), tag=str(self.ex.order_tag)[:20],
        )
        if self.ex.get("algo_id"):
            payload["algo_id"] = self.ex.algo_id
        if self.dry_run:
            return self._dry("place order", payload)
        order_id = self.kite.place_order(**payload)
        return {"status": "submitted", "ref": str(order_id), "message": f"Order {order_id} sent ({payload['variety']})."}

    def protect(self, symbol, qty, stop, target, last) -> dict:
        legs = protective_legs(stop, target, self.ex)
        if not legs:
            return {"ref": None, "message": ""}
        k = _KiteConstants
        sell = lambda px: {"transaction_type": k.TRANSACTION_TYPE_SELL, "quantity": int(qty),  # noqa: E731
                           "order_type": k.ORDER_TYPE_LIMIT, "product": k.PRODUCT_CNC, "price": px}
        if "target_trigger" in legs:
            payload = dict(trigger_type=k.GTT_TYPE_OCO, tradingsymbol=symbol, exchange=k.EXCHANGE_NSE,
                           trigger_values=[legs["stop_trigger"], legs["target_trigger"]], last_price=float(last),
                           orders=[sell(legs["stop_limit"]), sell(legs["target_limit"])])
        else:
            payload = dict(trigger_type=k.GTT_TYPE_SINGLE, tradingsymbol=symbol, exchange=k.EXCHANGE_NSE,
                           trigger_values=[legs["stop_trigger"]], last_price=float(last),
                           orders=[sell(legs["stop_limit"])])
        if self.dry_run:
            return {**self._dry("place GTT", payload), **legs}
        resp = self.kite.place_gtt(**payload)
        gid = str(resp.get("trigger_id", resp) if isinstance(resp, dict) else resp)
        return {"ref": gid, "message": f"GTT {gid} placed.", **legs}

    def remove_protection(self, ref, symbol) -> dict:
        if not ref or self.dry_run or str(ref).startswith("DRY"):
            return {"message": f"(dry run) would delete GTT {ref} for {symbol}"}
        self.kite.delete_gtt(int(ref))
        return {"message": f"GTT {ref} deleted."}

    def cancel_open_orders(self) -> dict:
        if self.dry_run:
            return {"message": "(dry run) would cancel all open indquant orders"}
        n = 0
        for o in self.kite.orders():
            if o.get("tag") == str(self.ex.order_tag)[:20] and o.get("status") in OPEN_STATUSES:
                self.kite.cancel_order(variety=o["variety"], order_id=o["order_id"])
                n += 1
        return {"message": f"Cancelled {n} open order(s). Protective GTTs were left in place."}

    def fills(self) -> list[dict]:
        if self.dry_run:
            return []
        out = []
        for o in self.kite.orders():
            if o.get("tag") == str(self.ex.order_tag)[:20] and o.get("status") == "COMPLETE":
                out.append({"ref": str(o["order_id"]), "qty": int(o.get("filled_quantity") or o["quantity"]),
                            "price": float(o["average_price"])})
        return out

    def triggered_protections(self) -> list[str]:
        if self.dry_run:
            return []
        return [str(g["id"]) for g in self.kite.get_gtts() if g.get("status") == "triggered"]


class _KiteConstants:
    """Kite Connect constant values, so dry runs work without the SDK installed."""
    VARIETY_REGULAR, VARIETY_AMO = "regular", "amo"
    EXCHANGE_NSE = "NSE"
    TRANSACTION_TYPE_BUY, TRANSACTION_TYPE_SELL = "BUY", "SELL"
    PRODUCT_CNC = "CNC"
    ORDER_TYPE_LIMIT = "LIMIT"
    GTT_TYPE_OCO, GTT_TYPE_SINGLE = "two-leg", "single"


def make_broker(cfg, kite=None) -> Broker:
    if cfg.execution.broker == "kite":
        return KiteBroker(cfg, kite)
    if cfg.execution.broker == "paper":
        return PaperBroker(cfg)
    return ManualBroker(cfg)


# ------------------------------------------------------------------ Kite login
def kite_login_url(cfg) -> str:
    from kiteconnect import KiteConnect

    return KiteConnect(api_key=os.environ[cfg.execution.kite_api_key_env]).login_url()


def kite_complete_login(cfg, request_token: str) -> None:
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=os.environ[cfg.execution.kite_api_key_env])
    session = kite.generate_session(request_token, api_secret=os.environ[cfg.execution.kite_api_secret_env])
    path = Path(cfg.execution.kite_token_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access_token": session["access_token"], "date": str(now_ist().date())}))
    os.chmod(path, 0o600)


def load_kite_token(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["access_token"] if d.get("date") == str(date.today()) else None
