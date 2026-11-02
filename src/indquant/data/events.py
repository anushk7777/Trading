"""Known corporate events (results dates, AGMs, record dates) -> entry blackout mask.

CSV format: ``symbol,date[,type]``. Earnings announcements cause gaps that blow through
stops; most discretionary swing traders avoid holding into them, so the swing sleeve
refuses new entries within ``event_blackout_days`` sessions before an event.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def blackout_mask(path: str | Path | None, dates: pd.DatetimeIndex, symbols: list[str], days_before: int,
                  days_after: int = 1) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    mask = pd.DataFrame(False, index=dates, columns=symbols)
    col = {s: j for j, s in enumerate(symbols)}
    arr = mask.to_numpy()
    for sym, d in zip(raw["symbol"].astype(str), raw["date"]):
        j = col.get(sym)
        if j is None:
            continue
        k = int(dates.searchsorted(d))
        lo, hi = max(0, k - days_before), min(len(dates), k + days_after + 1)
        arr[lo:hi, j] = True
    return pd.DataFrame(arr, index=dates, columns=symbols)
