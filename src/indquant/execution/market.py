"""NSE session clock and price rounding."""
from __future__ import annotations

import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
OPEN, CLOSE = time(9, 15), time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(d, holidays=()) -> bool:
    return d.weekday() < 5 and str(d) not in set(map(str, holidays))


def market_open(now: datetime | None = None, holidays=()) -> bool:
    now = now or now_ist()
    return is_trading_day(now.date(), holidays) and OPEN <= now.time() <= CLOSE


def round_tick(price: float, tick: float, side: str) -> float:
    """Round to the exchange tick in the direction that keeps the order conservative."""
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if side == "up" else math.floor(steps + 1e-9)
    return round(steps * tick, 2)
