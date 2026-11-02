"""Investable universe: index constituents, point-in-time membership and liquidity filters."""
from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT

log = logging.getLogger(__name__)

NSE_INDEX_FILES = {
    "nifty50": "ind_nifty50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
    "niftymidcap150": "ind_niftymidcap150list.csv",
    "niftysmallcap250": "ind_niftysmallcap250list.csv",
}
NSE_BASES = [
    "https://nsearchives.nseindia.com/content/indices/",
    "https://archives.nseindia.com/content/indices/",
]
FALLBACK_FILE = REPO_ROOT / "data" / "universe" / "nifty50_fallback.csv"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    cols = {c.lower(): c for c in df.columns}
    sym = cols.get("symbol")
    if sym is None:
        raise ValueError("universe file needs a 'Symbol' column")
    out = pd.DataFrame({"Symbol": df[sym].astype(str).str.strip()})
    ind = cols.get("industry") or cols.get("sector")
    out["Industry"] = df[ind].astype(str).str.strip() if ind else "Unknown"
    name = cols.get("company name") or cols.get("company")
    out["Company"] = df[name].astype(str).str.strip() if name else out["Symbol"]
    if "series" in cols:
        out = out[df[cols["series"]].astype(str).str.strip().isin(["EQ", "BE", "nan"]).to_numpy()]
    return out.drop_duplicates("Symbol").reset_index(drop=True)


def fetch_index_constituents(name: str, cache_dir: str | Path, refresh: bool = False) -> pd.DataFrame:
    """Download current constituents of an NSE index (cached). Falls back to bundled Nifty 50."""
    name = name.lower()
    if name not in NSE_INDEX_FILES:
        raise ValueError(f"Unknown index {name}. Choose from {sorted(NSE_INDEX_FILES)}")
    cache = Path(cache_dir) / f"universe_{name}.csv"
    if cache.exists() and not refresh:
        return _normalise(pd.read_csv(cache))
    for base in NSE_BASES:
        url = base + NSE_INDEX_FILES[name]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/csv,*/*"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            df = _normalise(pd.read_csv(io.StringIO(text)))
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
            log.info("Downloaded %d constituents of %s", len(df), name)
            return df
        except Exception as exc:  # network errors, NSE blocking, parse errors
            log.warning("Could not fetch %s from %s: %s", name, url, exc)
    log.warning("Falling back to bundled Nifty 50 list (%s). Results will cover far fewer stocks.", FALLBACK_FILE)
    return _normalise(pd.read_csv(FALLBACK_FILE))


def load_universe(cfg) -> pd.DataFrame:
    if cfg.data.universe == "file":
        if not cfg.data.universe_file:
            raise ValueError("data.universe=file requires data.universe_file")
        return _normalise(pd.read_csv(cfg.data.universe_file))
    return fetch_index_constituents(cfg.data.universe, cfg.data.cache_dir)


def load_membership(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Point-in-time membership from a CSV of (date, symbol) snapshots.

    Each date lists the full constituent set effective from that date until the next
    snapshot. This is how survivorship bias is removed: stocks that were in the index
    in 2016 but later dropped (or delisted) are still in the 2016 universe.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    snaps = sorted(raw["date"].unique())
    out = pd.DataFrame(False, index=dates, columns=symbols)
    for i, snap in enumerate(snaps):
        end = snaps[i + 1] if i + 1 < len(snaps) else None
        members = set(raw.loc[raw["date"] == snap, "symbol"].astype(str))
        cols = [s for s in symbols if s in members]
        mask = (dates >= snap) if end is None else ((dates >= snap) & (dates < end))
        out.loc[mask, cols] = True
    return out


def eligibility_mask(data, cfg) -> pd.DataFrame:
    """Boolean frame: stock is tradable/eligible on date t using information up to t only."""
    f = cfg.universe_filter
    close = data.close
    traded_value = close * data.volume
    adv = traded_value.rolling(int(f.adv_window), min_periods=max(5, int(f.adv_window) // 2)).mean()
    history = close.notna().cumsum()
    mask = (close >= float(f.min_price)) & (adv >= float(f.min_adv_inr)) & (history >= int(f.min_history_days))
    mask &= close.notna()
    if data.membership is not None:
        mask &= data.membership
    return mask.fillna(False).astype(bool)


def top_liquid(mask: pd.DataFrame, data, n: int, window: int = 63) -> pd.DataFrame:
    """Restrict an eligibility mask to the ``n`` most liquid stocks each day."""
    adv = (data.close * data.volume).rolling(window, min_periods=10).mean().where(mask)
    rank = adv.rank(axis=1, ascending=False, method="first")
    return (mask & (rank <= n)).fillna(False)


def adv_value(data, window: int = 21) -> pd.DataFrame:
    return (data.close * data.volume).rolling(window, min_periods=5).mean()


def daily_vol(data, window: int = 21) -> pd.DataFrame:
    ret = data.close.pct_change(fill_method=None)
    return ret.rolling(window, min_periods=10).std()


def annualise_vol(daily: pd.DataFrame | pd.Series):
    return daily * np.sqrt(252)
