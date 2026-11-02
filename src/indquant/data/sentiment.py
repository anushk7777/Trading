"""News-headline sentiment.

Two uses:
1. **Backtests** - only with a *point-in-time* sentiment file (symbol, timestamp, score) where
   the timestamp is when the headline was published. Headlines after 15:30 IST count for
   the next session. Without such history, sentiment is neutral in backtests (we never
   pretend to know old news we didn't record).
2. **Live** - RSS headlines from Indian business news feeds are matched to companies,
   scored with a finance word list (Loughran-McDonald style) and appended to a history
   file every run. After a few months that history can itself be tested like any other
   factor before trusting it beyond a simple veto.
"""
from __future__ import annotations

import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "gain", "gains",
    "record", "profit", "profits", "growth", "upgrade", "upgrades", "upgraded", "outperform", "buy", "strong",
    "robust", "win", "wins", "order", "orders", "contract", "approval", "approved", "expansion", "dividend",
    "bonus", "buyback", "rises", "rise", "higher", "boost", "boosts", "positive", "acquire", "acquires",
    "launch", "launches", "breakout", "bullish", "exceeds", "tops", "improves", "improved", "recovery",
}
NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "slump", "slumps", "crash", "crashes", "fall", "falls", "drop",
    "drops", "loss", "losses", "decline", "declines", "downgrade", "downgrades", "downgraded", "sell",
    "weak", "fraud", "probe", "raid", "raids", "penalty", "fine", "fined", "default", "defaults", "resigns",
    "resignation", "lawsuit", "ban", "banned", "sebi", "investigation", "pledge", "pledged", "warning",
    "cut", "cuts", "lower", "slips", "tumbles", "bearish", "concern", "concerns", "risk", "debt", "scam",
    "delay", "delayed", "halt", "halted", "recall", "strike", "shutdown", "insolvency", "writeoff",
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}
_SUFFIXES = re.compile(r"\b(ltd|limited|ltd\.|co|company|corporation|corp|india|industries|of|the|and|&)\b\.?", re.I)
IST_CUTOFF_HOUR = 15.5


def score_text(text: str) -> float:
    """Score in [-1, 1]: (pos - neg) / (pos + neg), with simple negation flipping."""
    words = re.findall(r"[a-z]+", text.lower())
    pos = neg = 0
    for k, w in enumerate(words):
        flip = k > 0 and words[k - 1] in NEGATORS
        if w in POSITIVE:
            neg, pos = (neg + 1, pos) if flip else (neg, pos + 1)
        elif w in NEGATIVE:
            pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def company_aliases(universe: pd.DataFrame) -> dict[str, list[str]]:
    """Search strings per symbol: the symbol itself and the cleaned company name."""
    out = {}
    for _, row in universe.iterrows():
        sym = str(row["Symbol"])
        name = _SUFFIXES.sub(" ", str(row.get("Company", sym)))
        name = re.sub(r"\s+", " ", name).strip()
        aliases = [sym] if len(sym) >= 3 else []
        if len(name) >= 4 and name.upper() != sym:
            aliases.append(name)
        out[sym] = aliases
    return out


def match_symbols(headline: str, aliases: dict[str, list[str]]) -> list[str]:
    hits = []
    for sym, names in aliases.items():
        for a in names:
            if re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", headline, flags=re.I if a != sym else 0):
                hits.append(sym)
                break
    return hits


def fetch_headlines(feeds: list[str], timeout: int = 15) -> pd.DataFrame:
    rows = []
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 indquant"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                root = ET.fromstring(resp.read())
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                pub = item.findtext("pubDate")
                try:
                    ts = parsedate_to_datetime(pub).astimezone(tz=None) if pub else pd.Timestamp.utcnow()
                except (TypeError, ValueError):
                    ts = pd.Timestamp.utcnow()
                rows.append({"timestamp": pd.Timestamp(ts).tz_convert("Asia/Kolkata").tz_localize(None),
                             "headline": title, "source": url})
        except Exception as exc:
            log.warning("News feed failed (%s): %s", url, exc)
    return pd.DataFrame(rows, columns=["timestamp", "headline", "source"])


def update_history(history_file: str | Path, feeds: list[str], universe: pd.DataFrame) -> pd.DataFrame:
    """Fetch, match, score and append new headlines. Returns the rows added."""
    heads = fetch_headlines(feeds)
    if heads.empty:
        return heads
    aliases = company_aliases(universe)
    rows = []
    for _, h in heads.iterrows():
        for sym in match_symbols(h["headline"], aliases):
            rows.append({"symbol": sym, "timestamp": h["timestamp"], "score": score_text(h["headline"]),
                         "headline": h["headline"]})
    new = pd.DataFrame(rows, columns=["symbol", "timestamp", "score", "headline"])
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["timestamp"])
        merged = pd.concat([old, new]).drop_duplicates(["symbol", "headline"], keep="first")
        added = len(merged) - len(old)
    else:
        merged, added = new, len(new)
    merged.to_csv(path, index=False)
    log.info("News: %d headlines fetched, %d new company matches stored", len(heads), added)
    return new


def sentiment_frame(path: str | Path | None, dates: pd.DatetimeIndex, symbols: list[str],
                    halflife_days: float = 3.0) -> pd.DataFrame | None:
    """Point-in-time decayed sentiment per (date, symbol); 0 = no news."""
    if not path or not Path(path).exists():
        return None
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    if raw.empty:
        return None
    ts = pd.to_datetime(raw["timestamp"])
    after_close = (ts.dt.hour + ts.dt.minute / 60.0) >= IST_CUTOFF_HOUR
    day = ts.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    # map each publication day to the first trading session on/after it
    pos = dates.searchsorted(day)
    valid = pos < len(dates)
    raw = raw.loc[valid.to_numpy() if hasattr(valid, "to_numpy") else valid].copy()
    raw["session"] = dates[pos[valid]]
    daily = raw.pivot_table(index="session", columns="symbol", values="score", aggfunc="mean")
    daily = daily.reindex(index=dates, columns=symbols)
    alpha = 1 - np.exp(np.log(0.5) / max(halflife_days, 1e-6))
    # decayed average of recent news; days without news decay toward 0
    filled = daily.fillna(0.0)
    count = daily.notna().astype(float)
    num = filled.ewm(alpha=alpha, adjust=False).mean()
    den = count.ewm(alpha=alpha, adjust=False).mean()
    out = (num / den.replace(0, np.nan)) * den.clip(upper=1.0)
    return out.fillna(0.0)
