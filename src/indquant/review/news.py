"""Company news for the AI reviewer, strictly as of the analysis date.

Point-in-time rules (borrowed from TradingAgents' date_window): only items published
within [as_of - lookback, as_of] are kept; undated items are kept only when the review
date is today, since a historical review cannot prove an undated item is not from the
future.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..data.sentiment import fetch_headlines

log = logging.getLogger(__name__)
_SUFFIX = re.compile(r"\b(ltd|limited|ltd\.)\b\.?", re.I)


def google_news_url(company: str, query_suffix: str, lookback_days: int) -> str:
    q = f'"{_SUFFIX.sub("", company).strip()}" {query_suffix} when:{lookback_days}d'
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})


def in_window(ts, as_of: pd.Timestamp, lookback_days: int) -> bool:
    if ts is None or pd.isna(ts):
        return as_of.normalize() >= pd.Timestamp(datetime.now().date())
    ts = pd.Timestamp(ts)
    return as_of.normalize() - timedelta(days=lookback_days) <= ts < as_of.normalize() + timedelta(days=1)


def gather_news(symbol: str, company: str, as_of, cfg, history_file: str | None = None,
                fetch=fetch_headlines) -> list[dict]:
    rc = cfg.reviewer
    as_of = pd.Timestamp(as_of)
    lookback = int(rc.lookback_days)
    items: list[dict] = []
    try:
        df = fetch([google_news_url(company or symbol, rc.news_query_suffix, lookback)])
        for _, r in df.iterrows():
            items.append({"timestamp": r["timestamp"], "title": r["headline"], "source": "news.google.com"})
    except Exception as exc:  # news is best effort; the reviewer reports thin coverage
        log.warning("News fetch failed for %s: %s", symbol, exc)
    path = history_file or cfg.sentiment.history_file
    if path and Path(path).exists():
        hist = pd.read_csv(path)
        hist = hist[hist["symbol"].astype(str) == symbol]
        for _, r in hist.iterrows():
            items.append({"timestamp": pd.to_datetime(r["timestamp"]), "title": r["headline"], "source": "rss history"})
    seen, out = set(), []
    for it in sorted(items, key=lambda x: pd.Timestamp(x["timestamp"]) if x["timestamp"] is not None
                     else pd.Timestamp.max, reverse=True):
        key = re.sub(r"\W+", " ", str(it["title"]).lower()).strip()[:120]
        if key in seen or not in_window(it["timestamp"], as_of, lookback):
            continue
        seen.add(key)
        out.append(it)
    return out[: int(rc.max_headlines)]
