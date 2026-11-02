"""Account-level risk governor: circuit breakers that sit *above* every strategy.

* Drawdown halt (kill switch): when a sleeve's drawdown exceeds ``max_drawdown_halt`` all
  new entries stop and the core sleeve is cut to ``halt_exposure``. Trading resumes only
  once the drawdown recovers below ``resume_drawdown`` (hysteresis avoids flip-flopping).
* Profit lock ("trap doors"): once equity has grown ``profit_lock_activation`` above its
  base, a trailing floor at ``peak * (1 - profit_lock_trail)`` is armed and only ratchets
  up. Falling through it de-risks the sleeve for ``profit_lock_cooldown_days`` sessions;
  afterwards the base resets to current equity, so the lock must be re-earned (this avoids
  a "trap" where reduced exposure can never climb back above the floor).
* Loss-streak brake: after ``loss_streak`` consecutive losing trades, position sizes are
  scaled by ``loss_streak_size_mult`` for ``loss_streak_cooldown_days`` sessions.

State lives in ``account.meta`` so it survives across paper-trading runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GovernorDecision:
    allow_new_entries: bool
    exposure_mult: float
    size_mult: float
    reasons: list[str]


def evaluate(account, equity: float, date, gcfg) -> GovernorDecision:
    meta = account.meta
    if not gcfg or not gcfg.enabled:
        return GovernorDecision(True, 1.0, 1.0, [])
    start = float(meta.setdefault("initial_equity", equity))
    peak = max(float(account.peak_equity or equity), equity)
    dd = 1 - equity / peak if peak > 0 else 0.0
    reasons: list[str] = []
    allow, exp_mult, size_mult = True, 1.0, 1.0

    # 1) drawdown kill switch with hysteresis
    halted = bool(meta.get("halted", False))
    if not halted and dd >= float(gcfg.max_drawdown_halt):
        halted = True
        meta["halt_events"] = int(meta.get("halt_events", 0)) + 1
    elif halted and dd <= float(gcfg.resume_drawdown):
        halted = False
    meta["halted"] = halted
    if halted:
        allow = False
        exp_mult *= float(gcfg.halt_exposure)
        reasons.append(f"drawdown halt ({dd:.1%} from peak)")

    # 2) profit lock / trailing equity floor ("trap door")
    date = pd.Timestamp(date)
    tripped_until = meta.get("lock_tripped_until")
    if tripped_until and date <= pd.Timestamp(tripped_until):
        exp_mult *= float(gcfg.profit_lock_exposure)
        size_mult *= float(gcfg.profit_lock_exposure)
        reasons.append(f"profit lock tripped - de-risked until {tripped_until}")
    else:
        if tripped_until:  # cool-off over: re-base so the lock must be re-earned with new gains
            meta.pop("lock_tripped_until", None)
            meta["lock_base"] = equity
            meta["lock_peak"] = equity
            meta["equity_floor"] = 0.0
        base = float(meta.setdefault("lock_base", start))
        lock_peak = max(float(meta.get("lock_peak", base)), equity)
        meta["lock_peak"] = lock_peak
        if lock_peak >= base * (1 + float(gcfg.profit_lock_activation)):
            floor = max(float(meta.get("equity_floor", 0.0)), lock_peak * (1 - float(gcfg.profit_lock_trail)))
            meta["equity_floor"] = floor
            if equity < floor:
                until = date + pd.tseries.offsets.BDay(int(gcfg.profit_lock_cooldown_days))
                meta["lock_tripped_until"] = str(until.date())
                meta["lock_trips"] = int(meta.get("lock_trips", 0)) + 1
                exp_mult *= float(gcfg.profit_lock_exposure)
                size_mult *= float(gcfg.profit_lock_exposure)
                reasons.append(f"profit lock tripped (equity {equity:,.0f} < floor {floor:,.0f})")

    # 3) loss-streak brake
    streak = int(meta.get("loss_streak", 0))
    until = pd.Timestamp(meta.get("streak_brake_until", "1900-01-01"))
    if streak >= int(gcfg.loss_streak) and until < date:
        until = date + pd.tseries.offsets.BDay(int(gcfg.loss_streak_cooldown_days))
        meta["streak_brake_until"] = str(until.date())
        meta["loss_streak"] = 0
    if date <= until:
        size_mult *= float(gcfg.loss_streak_size_mult)
        reasons.append("loss-streak brake")
    return GovernorDecision(allow, exp_mult, size_mult, reasons)
