"""Fail-safe data checks run before any live signal is produced.

Trading on bad data is worse than not trading. Errors block signal generation (unless the
user explicitly forces it); warnings are printed in the signal report.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_SETTLED = time(16, 0)


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    message: str


def drop_incomplete_bar(data, now: datetime | None = None):
    """During market hours the latest bar from free feeds is still forming - drop it."""
    now = now or datetime.now(IST)
    last = data.dates[-1]
    if last.date() == now.date() and now.time() < MARKET_CLOSE_SETTLED:
        return data.truncate(data.dates[-2]), True
    return data, False


def check_data(data, cfg, now: datetime | None = None) -> list[Issue]:
    h = cfg.healthcheck
    issues: list[Issue] = []
    dates = data.dates
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        issues.append(Issue("error", "Trading calendar is not strictly increasing (duplicate or unsorted dates)."))
    now = now or datetime.now(IST)
    last = dates[-1]
    stale = int(np.busday_count(last.date(), now.date()))
    if data.source != "synthetic" and stale > int(h.max_stale_days):
        issues.append(Issue("error", f"Data is stale: last bar {last.date()} is {stale} business days old."))
    close_last = data.close.iloc[-1]
    if data.close.shape[1]:
        expected = data.close.iloc[-20:].notna().any()
        coverage = close_last[expected].notna().mean() if expected.any() else 0.0
        if coverage < float(h.min_coverage):
            issues.append(Issue("error", f"Only {coverage:.0%} of active symbols have a price on {last.date()}."))
    if pd.isna(data.benchmark.iloc[-1]) or pd.isna(data.index.iloc[-1]):
        issues.append(Issue("error", "Benchmark / regime index has no value on the last date."))
    if len(dates) > 1:
        r = (data.close.iloc[-1] / data.close.iloc[-2] - 1).abs()
        extreme = (r > 0.20).mean()
        if extreme > float(h.max_extreme_moves):
            issues.append(Issue("error", f"{extreme:.0%} of stocks moved more than 20% on the last day - "
                                         "likely unadjusted corporate actions or bad data."))
        elif (r > 0.20).any():
            names = ", ".join(r[r > 0.20].index[:10])
            issues.append(Issue("warning", f"Large 1-day moves (>20%): {names}. Check for splits/bonuses."))
    zero_vol = ((data.volume.iloc[-1] <= 0) & close_last.notna()).sum()
    if zero_vol:
        issues.append(Issue("warning", f"{zero_vol} symbols show zero volume on the last day."))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)
