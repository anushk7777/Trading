"""Yahoo Finance loader for NSE stocks (``SYMBOL.NS``) with a per-symbol parquet cache.

Prices are downloaded with ``auto_adjust=True`` (split + dividend adjusted). Because
adjustments rewrite history whenever a new dividend is paid, the cache is refreshed by
re-downloading full history rather than appending.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def yahoo_symbol(symbol: str) -> str:
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("&", "_and_").replace("/", "_")
    return cache_dir / f"{safe}.parquet"


def _fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _extract(raw: pd.DataFrame, ysym: str, single: bool) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    df = raw
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        lv1 = raw.columns.get_level_values(1)
        if ysym in set(lv0):
            df = raw[ysym]
        elif ysym in set(lv1):
            df = raw.xs(ysym, axis=1, level=1)
        elif single:
            df = raw.droplevel(1, axis=1) if set(OHLCV) & set(lv0) else raw.droplevel(0, axis=1)
        else:
            return None
    cols = [c for c in OHLCV if c in df.columns]
    if "Close" not in cols:
        return None
    df = df[cols].dropna(how="all")
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_prices(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    max_age_hours: float = 12,
    refresh: bool = False,
    chunk: int = 40,
    adjust_dividends: bool = True,
    quiet: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV frame}. Symbols without data are skipped with a warning.

    ``adjust_dividends=False`` returns prices adjusted for splits/bonuses only (comparable with
    exchange prices, which also exclude dividends)."""
    import yfinance as yf

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    todo = []
    suffix = "" if adjust_dividends else "__split_only"
    for sym in symbols:
        path = _cache_path(cache, sym + suffix)
        if not refresh and _fresh(path, max_age_hours):
            out[sym] = pd.read_parquet(path)
        else:
            todo.append(sym)

    missing: list[str] = []
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        ysyms = [yahoo_symbol(s) for s in batch]
        log.info("Downloading %d symbols from Yahoo (%d/%d)", len(batch), i + len(batch), len(todo))
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    tickers=ysyms, start=start, end=end, auto_adjust=adjust_dividends, actions=False,
                    group_by="ticker", threads=True, progress=False,
                )
                break
            except Exception as exc:  # network hiccups / rate limiting
                log.warning("Yahoo download failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** (attempt + 1))
        for sym, ysym in zip(batch, ysyms):
            df = _extract(raw, ysym, single=len(batch) == 1)
            if df is None:
                stale = _cache_path(cache, sym + suffix)
                if stale.exists():
                    log.warning("No fresh data for %s; using stale cache", sym)
                    out[sym] = pd.read_parquet(stale)
                else:
                    (log.debug if quiet else log.warning)("No data for %s (%s) - skipped", sym, ysym)
                    missing.append(sym)
                continue
            df.to_parquet(_cache_path(cache, sym + suffix))
            out[sym] = df
    if quiet and missing:
        log.info("Yahoo has no data for %d of %d symbols (mostly delisted) - skipped", len(missing), len(symbols))
    if start:
        out = {k: v.loc[pd.Timestamp(start):] for k, v in out.items()}
    if end:
        out = {k: v.loc[: pd.Timestamp(end)] for k, v in out.items()}
    return out
