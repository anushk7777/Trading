"""What the backtest promised, and whether live trading is keeping that promise.

After each backtest the distribution of h-day returns is estimated by block bootstrap and
saved. Live (desk or paper) equity is then compared with it: falling below the 5th
percentile over a horizon means live results are worse than 95% of what the backtest
implied - the classic early warning that an edge has decayed or never existed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126, 252)


def build(daily_returns: pd.Series, horizons=HORIZONS, reps: int = 2000, block: int = 20, seed: int = 0) -> dict:
    r = daily_returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        if len(r) < h * 2:
            continue
        sims = np.empty(reps)
        for k in range(reps):
            idx = []
            while len(idx) < h:
                start = rng.integers(0, len(r))
                idx.extend((start + np.arange(min(block, h - len(idx)))) % len(r))
            sims[k] = np.prod(1 + r[idx]) - 1
        out[str(h)] = {q: float(np.quantile(sims, p)) for q, p in (("p05", .05), ("p25", .25), ("p50", .5),
                                                                    ("p75", .75), ("p95", .95))}
    return out


def save(path, quantiles: dict, meta: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), **meta,
                             "quantiles": quantiles}, indent=2, default=str))


def load(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def check(live_equity: pd.Series, expectations: dict | None) -> dict:
    """Compare the longest available live horizon with the backtest's distribution."""
    if not expectations or live_equity is None or len(live_equity.dropna()) < 22:
        return {"status": "insufficient", "message": "Not enough live history or no saved backtest expectations yet."}
    eq = live_equity.dropna()
    q = expectations["quantiles"]
    usable = [int(h) for h in q if int(h) < len(eq)]
    if not usable:
        return {"status": "insufficient", "message": "Not enough live history yet."}
    h = max(usable)
    live = float(eq.iloc[-1] / eq.iloc[-1 - h] - 1)
    band = q[str(h)]
    status = "below" if live < band["p05"] else ("above" if live > band["p95"] else "normal")
    msg = {"below": f"DRIFT ALERT: live {h}-day return {live:.1%} is below the backtest's 5th percentile "
                    f"({band['p05']:.1%}). The edge may have decayed - consider pausing and re-validating.",
           "above": f"Live {h}-day return {live:.1%} is above the backtest's 95th percentile ({band['p95']:.1%}) - "
                    "unusually good luck or a changed regime; do not raise risk because of it.",
           "normal": f"Live {h}-day return {live:.1%} is within the backtest's range "
                     f"(5th-95th pct {band['p05']:.1%} to {band['p95']:.1%})."}[status]
    return {"status": status, "horizon": h, "live_return": live, **band, "message": msg}
