"""Survivorship-free NSE data from the exchange's own daily bhavcopy files.

Why: Yahoo only has stocks that still trade, and a backtest on today's NIFTY 500 list
never sees the companies that collapsed or were delisted - results are overstated.
Every listed equity appears in the bhavcopy of every day it traded, including ones that
no longer exist.

Pipeline
1. download   - daily CM bhavcopy zips (legacy format until 2024-07-05, UDiFF after),
                cached raw; holidays (404) remembered so they are not retried.
2. parse      - column names are matched by alias, so both formats load into one schema.
3. stitch     - symbol renames (same ISIN, old symbol stops the day before the new one
                starts) are merged under the latest symbol.
4. adjust     - splits/bonuses are back-adjusted with the factor the exchange itself
                applies to the previous close (PREVCLOSE_t / CLOSE_t-1); a clean-ratio
                overnight-gap detector is the fallback; a user corporate-actions file
                overrides both. Optional dividends file makes prices total-return.
                Every adjustment is written to a CSV for review.
5. universe   - point-in-time membership: each month, the top-N stocks by trailing
                traded value among those listed >= 1 year. No index list from today.
"""
from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UDIFF_FROM = date(2024, 7, 8)
BASE = "https://nsearchives.nseindia.com/content"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ALIASES = {
    "symbol": ["SYMBOL", "TCKRSYMB"],
    "series": ["SERIES", "SCTYSRS"],
    "open": ["OPEN", "OPNPRIC", "OPEN_PRICE"],
    "high": ["HIGH", "HGHPRIC", "HIGH_PRICE"],
    "low": ["LOW", "LWPRIC", "LOW_PRICE"],
    "close": ["CLOSE", "CLSPRIC", "CLOSE_PRICE"],
    "prev_close": ["PREVCLOSE", "PRVSCLSGPRIC", "PREV_CLOSE"],
    "volume": ["TOTTRDQTY", "TTLTRADGVOL", "TTL_TRD_QNTY"],
    "value": ["TOTTRDVAL", "TTLTRFVAL"],
    "date": ["TIMESTAMP", "TRADDT", "DATE1"],
    "isin": ["ISIN"],
    "instrument": ["FININSTRMTP"],
}
# Weekend sessions NSE held (Budget days, Diwali Muhurat, special live sessions). A missing session
# makes every stock's PREVCLOSE differ from our last close the next day, which would look like
# thousands of tiny "splits". Unknown future ones are caught by the market-wide check below.
SPECIAL_SESSIONS = {date.fromisoformat(d) for d in (
    "2015-02-28", "2016-10-30", "2019-10-27", "2020-02-01", "2020-11-14", "2023-11-12",
    "2024-01-20", "2024-03-02", "2024-05-18", "2025-02-01", "2026-02-01", "2026-11-08")}
CLEAN_RATIOS = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1 / 5, 2 / 5, 1 / 10, 1 / 2.5, 1 / 20])


# ------------------------------------------------------------------------- download
def bhavcopy_url(d: date) -> str:
    if d >= UDIFF_FROM:
        return f"{BASE}/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"{BASE}/historical/EQUITIES/{d.year}/{mon}/cm{d:%d}{mon}{d.year}bhav.csv.zip"


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*", "Referer": "https://www.nseindia.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


MAX_CONSECUTIVE_FAILURES = 15


def download_range(start: date, end: date, raw_dir: str | Path, fetch=_http_get, pause: float = 0.35,
                   progress_every: float = 15.0) -> dict:
    """Fetch every weekday in [start, end] not already cached. Returns counts.

    Prints progress every ``progress_every`` seconds, writes each file atomically (Ctrl+C is safe:
    re-running resumes) and gives up early when every request fails (blocked network)."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    missing_file = raw / "missing_days.txt"
    missing = set(missing_file.read_text().split()) if missing_file.exists() else set()
    today = date.today()
    stats = {"downloaded": 0, "cached": 0, "holidays": 0, "failed": 0}
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 or d in SPECIAL_SESSIONS]
    todo = [d for d in days if not (raw / f"{d:%Y%m%d}.csv.zip").exists()
            and not (f"{d:%Y%m%d}" in missing and (today - d).days > 7)]
    stats["cached"] = sum((raw / f"{d:%Y%m%d}.csv.zip").exists() for d in days)
    stats["holidays"] = len(days) - stats["cached"] - len(todo)
    if not todo:
        return stats
    log.info("Downloading %d NSE daily price files (%s to %s) into %s. %s", len(todo), todo[0], todo[-1], raw,
             ("First time: about 30-60 minutes. " if len(todo) > 500 else "")
             + "Safe to stop with Ctrl+C; run the same command again to resume.")
    t0 = last = time.monotonic()
    streak = 0
    for n, d in enumerate(todo, 1):
        key = f"{d:%Y%m%d}"
        path = raw / f"{key}.csv.zip"
        try:
            body = fetch(bhavcopy_url(d))
            tmp = path.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(path)                                 # never leaves a half-written file behind
            stats["downloaded"] += 1
            streak = 0
            time.sleep(pause)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                               # market holiday: no file that day
                missing.add(key)
                stats["holidays"] += 1
                streak = 0
            else:
                stats["failed"] += 1
                streak += 1
                log.warning("Bhavcopy %s: HTTP %s", d, exc.code)
        except Exception as exc:  # network trouble: keep going, retry next run
            stats["failed"] += 1
            streak += 1
            log.warning("Bhavcopy %s failed: %s", d, exc)
        if streak >= MAX_CONSECUTIVE_FAILURES:
            log.error("Stopping: the last %d downloads all failed, so NSE is probably blocking this network "
                      "(office/college Wi-Fi, VPN or firewall) or is down. Try another network or later, "
                      "or use `--set data.source=yahoo`. Files already downloaded are kept.", streak)
            break
        now = time.monotonic()
        if now - last >= progress_every or n == len(todo):
            rate = (now - t0) / n
            left = (len(todo) - n) * rate / 60
            log.info("  %s: %d/%d files done (%d holidays, %d failed), about %.0f min left",
                     d, n, len(todo), stats["holidays"], stats["failed"], left)
            last = now
    missing_file.write_text("\n".join(sorted(missing)))
    return stats


# ------------------------------------------------------------------------- parse
def _read_any(path_or_bytes) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            data = zf.read(name)
    return pd.read_csv(io.BytesIO(data), dtype=str)


def _parse_dates(col: pd.Series) -> pd.Series:
    """Explicit formats: UDiFF uses ISO (2024-07-08), legacy uses 02-JAN-2023. Never guess day-first on ISO."""
    sample = col.dropna().iloc[0] if col.notna().any() else ""
    for fmt, pattern in (("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}"), ("%d-%b-%Y", r"^\d{1,2}-[A-Za-z]{3}-\d{4}"),
                         ("%d-%m-%Y", r"^\d{2}-\d{2}-\d{4}"), ("%d %b %Y", r"^\d{1,2} [A-Za-z]{3} \d{4}")):
        if pd.Series([sample]).str.match(pattern).iloc[0]:
            return pd.to_datetime(col.str.slice(0, 11).str.strip(), format=fmt, errors="coerce").dt.normalize()
    return pd.to_datetime(col, errors="coerce").dt.normalize()


def parse_bhavcopy(path_or_bytes, series=("EQ", "BE")) -> pd.DataFrame:
    """One day's bhavcopy (either format) -> normalised long frame."""
    raw = _read_any(path_or_bytes)
    cols = {c.strip().upper(): c for c in raw.columns}
    out = {}
    for name, aliases in ALIASES.items():
        src = next((cols[a] for a in aliases if a in cols), None)
        if src is not None:
            out[name] = raw[src].astype(str).str.strip()
    need = {"symbol", "series", "open", "high", "low", "close", "volume", "date"}
    if not need <= set(out):
        raise ValueError(f"unrecognised bhavcopy format; columns: {list(raw.columns)[:12]}")
    df = pd.DataFrame(out)
    if "instrument" in df:
        df = df[df["instrument"].isin(["STK", "nan"])]
    df = df[df["series"].isin(list(series))].copy()
    for c in ("open", "high", "low", "close", "prev_close", "volume", "value"):
        if c in df:
            df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    if "value" not in df:
        df["value"] = df["close"] * df["volume"]
    if "prev_close" not in df:
        df["prev_close"] = np.nan
    if "isin" not in df:
        df["isin"] = ""
    df["date"] = _parse_dates(df["date"])
    df = df.dropna(subset=["date", "close"])
    # one row per symbol and day (EQ preferred over BE if both exist)
    df["_rank"] = df["series"].map({s: i for i, s in enumerate(series)})
    df = df.sort_values("_rank").drop_duplicates(["date", "symbol"]).drop(columns="_rank")
    return df[["date", "symbol", "series", "isin", "open", "high", "low", "close", "prev_close", "volume", "value"]]


def build_long(raw_dir: str | Path, cache: str | Path | None = None, series=("EQ", "BE")) -> pd.DataFrame:
    """Parse every cached zip (incrementally, using a parquet cache of parsed days)."""
    raw = Path(raw_dir)
    files = sorted(raw.glob("*.csv.zip"))
    parsed = pd.DataFrame()
    done: set = set()
    if cache and Path(cache).exists():
        parsed = pd.read_parquet(cache)
        done = set(parsed["file"].unique())
    new = []
    for f in files:
        if f.name in done:
            continue
        try:
            day = parse_bhavcopy(f, series)
            day["file"] = f.name
            new.append(day)
        except Exception as exc:
            log.warning("Skipping unreadable bhavcopy %s: %s", f.name, exc)
    if new:
        parsed = pd.concat([parsed, *new], ignore_index=True)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            parsed.to_parquet(cache)
    return parsed


# ------------------------------------------------------------------------- stitch & adjust
def stitch_renames(long: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """Merge symbol renames: same ISIN, old symbol's last day precedes the new symbol's first day."""
    spans = long.groupby("symbol").agg(first=("date", "min"), last=("date", "max"), isin=("isin", "last"))
    renames = []
    for isin, grp in spans[spans["isin"].astype(bool) & (spans["isin"] != "nan")].groupby("isin"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("first")
        chain = list(grp.index)
        for old, new in zip(chain[:-1], chain[1:]):
            if grp.loc[old, "last"] < grp.loc[new, "first"]:
                renames.append((old, new, str(grp.loc[new, "first"].date())))
    mapping = {}
    for old, new, _ in renames:
        mapping[old] = new
    for old in list(mapping):  # resolve chains A->B->C
        tgt = mapping[old]
        while tgt in mapping:
            tgt = mapping[tgt]
        mapping[old] = tgt
    if mapping:
        long = long.copy()
        long["symbol"] = long["symbol"].replace(mapping)
        long = long.drop_duplicates(["date", "symbol"], keep="last")
    return long, renames


def _wide(long: pd.DataFrame, col: str) -> pd.DataFrame:
    return long.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()


def _load_overrides(path) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["symbol", "ex_date", "factor"])
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.normalize()
    return df[["symbol", "ex_date", "factor"]]


def detect_actions(close: pd.DataFrame, prev_close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame,
                   threshold: float = 0.03, gap_heuristic: bool = True, stale_sessions: int = 5) -> pd.DataFrame:
    """Split / bonus events as (symbol, ex_date, factor, source).

    factor multiplies every price BEFORE ex_date (0.5 for a 1:1 bonus or 2:1 split).

    The exchange's PREVCLOSE is adjusted on ex-dates, so prev_close / our last close != 1 marks a
    corporate action - except when (a) a whole session is missing from our files (most stocks then
    disagree by that session's move), or (b) the stock was away for weeks (e.g. traded in a series we
    don't load), where the ratio is mostly price drift. Those cases need a much bigger, cleaner ratio.
    The overnight-gap fallback looks for >30% opening gaps at clean ratios: NSE price bands (at most
    20%) make such an open impossible for a real move - except for a stock returning after weeks away,
    which is excluded so a genuine loss is never "adjusted" away.
    """
    last_close = close.ffill().shift(1)            # last traded close before each day
    rows = []
    ratio = (prev_close / last_close).where(close.notna())
    dev = (ratio - 1).abs()
    seen = ratio.notna().sum(axis=1)
    differ = (dev > 0.003).sum(axis=1) / seen.where(seen >= 20)
    missing_session = (differ > 0.3).to_numpy()
    for d in np.nonzero(missing_session)[0]:
        log.warning("Bhavcopy %s: %.0f%% of stocks have a different previous close - the session before it "
                    "is missing from data/nse/raw (special weekend session?). Run `indquant fetch`.",
                    close.index[d].date(), 100 * differ.iloc[d])
    obs = close.notna().to_numpy()
    last_seen = np.where(obs, np.arange(len(close))[:, None], np.nan)
    last_seen = pd.DataFrame(last_seen, index=close.index, columns=close.columns).ffill().shift(1)
    away = (np.arange(len(close))[:, None] - last_seen.to_numpy()) > stale_sessions
    hits = dev > threshold
    arr, dev_arr = ratio.to_numpy(), dev.to_numpy()
    for (d, s) in zip(*np.nonzero(hits.to_numpy())):
        r = float(arr[d, s])
        if missing_session[d] and dev_arr[d, s] < 0.25:
            continue                                  # just the missing session's price move
        if away[d, s]:
            k = int(np.argmin(np.abs(CLEAN_RATIOS - r)))
            if abs(CLEAN_RATIOS[k] - r) / CLEAN_RATIOS[k] >= 0.03:
                continue                              # drift while the stock was away, not an action
        rows.append((close.columns[s], close.index[d], r, "exchange_prev_close"))
    if gap_heuristic:
        gap = (open_ / last_close).where(close.notna() & ~hits & ~pd.DataFrame(away, index=close.index,
                                                                                  columns=close.columns))
        vol_ratio = volume / volume.rolling(20, min_periods=5).median().shift(1)
        arr = gap.to_numpy()
        with np.errstate(invalid="ignore"):
            cand = np.abs(arr - 1) > 0.3
        for (d, s) in zip(*np.nonzero(cand)):
            g = arr[d, s]
            k = int(np.argmin(np.abs(CLEAN_RATIOS - g)))
            vr = vol_ratio.iat[d, s]
            if abs(CLEAN_RATIOS[k] - g) / CLEAN_RATIOS[k] < 0.03 and (np.isnan(vr) or vr > 1.3):
                rows.append((close.columns[s], close.index[d], float(CLEAN_RATIOS[k]), "gap_heuristic"))
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "factor", "source"])


def adjust_prices(frames: dict[str, pd.DataFrame], actions: pd.DataFrame,
                  dividends: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Back-adjust OHLC (and volume) for the given actions; optional dividends make it total-return."""
    close = frames["close"]
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    events = actions[["symbol", "ex_date", "factor"]].copy()
    if dividends is not None and len(dividends):
        last_close = close.ffill().shift(1)
        div_rows = []
        for _, r in dividends.iterrows():
            if r["symbol"] in close.columns:
                k = int(close.index.searchsorted(r["ex_date"]))
                if 0 < k < len(close.index):
                    pc = last_close[r["symbol"]].iloc[k]
                    if pc and not np.isnan(pc):
                        div_rows.append((r["symbol"], close.index[k], 1 - float(r["amount"]) / pc))
        events = pd.concat([events, pd.DataFrame(div_rows, columns=["symbol", "ex_date", "factor"])])
    for _, e in events.iterrows():
        if e["symbol"] not in factor.columns or not 0 < e["factor"] < 10:
            continue
        mask = factor.index < e["ex_date"]
        factor.loc[mask, e["symbol"]] *= float(e["factor"])
    out = {k: frames[k] * factor for k in ("open", "high", "low", "close")}
    out["volume"] = frames["volume"] / factor
    out["value"] = frames["value"]
    return out


def point_in_time_universe(value: pd.DataFrame, close: pd.DataFrame, size: int, lookback: int,
                           min_listing_days: int) -> pd.DataFrame:
    """Monthly top-N by trailing traded value among sufficiently seasoned listings."""
    adv = value.rolling(lookback, min_periods=min(lookback, max(20, lookback // 3))).mean()
    seasoned = close.notna().cumsum() >= min_listing_days
    adv = adv.where(seasoned & close.notna())
    month_end = pd.Series(close.index, index=close.index).groupby(close.index.to_period("M")).transform("max")
    snap_days = close.index[close.index == month_end.to_numpy()]
    rank = adv.loc[snap_days].rank(axis=1, ascending=False, method="first")
    snap = (rank <= size)
    # effective from the next session after the month-end snapshot (no same-day look-ahead)
    member = snap.reindex(close.index).shift(1).ffill().fillna(False).astype(bool)
    return member


# ------------------------------------------------------------------------- entry point
def load_nse_panel(cfg, refresh: bool = False, fetch=_http_get) -> dict:
    """Download (if needed) and build adjusted panels + membership. Returns dict of frames and logs."""
    n = cfg.nse
    raw_dir = Path(n.raw_dir)
    start = pd.Timestamp(cfg.general.start).date()
    end = pd.Timestamp(cfg.general.end).date() if cfg.general.end else date.today()
    cached = sorted(raw_dir.glob("*.csv.zip"))
    if refresh or not cached:
        stats = download_range(start, end, raw_dir, fetch=fetch, pause=float(n.request_pause))
        log.info("Bhavcopy download: %s", stats)
    else:  # daily use: only fetch the days after the newest cached file
        newest = pd.Timestamp(cached[-1].name[:8]).date()
        if newest < end:
            stats = download_range(newest + timedelta(days=1), end, raw_dir, fetch=fetch,
                                   pause=float(n.request_pause))
            log.info("Bhavcopy update: %s", stats)
    long = build_long(raw_dir, raw_dir.parent / "parsed_long.parquet", tuple(n.series))
    if long.empty:
        raise RuntimeError(f"No bhavcopy files in {raw_dir}. Run `indquant --source nse fetch`.")
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] <= pd.Timestamp(end))]
    long, renames = stitch_renames(long)
    frames = {c: _wide(long, c) for c in ("open", "high", "low", "close", "prev_close", "volume", "value")}
    idx, cols = frames["close"].index, frames["close"].columns
    frames = {k: v.reindex(index=idx, columns=cols) for k, v in frames.items()}

    detected = detect_actions(frames["close"], frames["prev_close"], frames["open"], frames["volume"],
                              float(n.ca_threshold), bool(n.apply_gap_heuristic))
    overrides = _load_overrides(n.corporate_actions_file)
    if len(overrides):
        key = set(zip(overrides["symbol"], overrides["ex_date"]))
        detected = detected[[(s, d) not in key for s, d in zip(detected["symbol"], detected["ex_date"])]]
        actions = pd.concat([detected, overrides.assign(source="user_file")], ignore_index=True)
    else:
        actions = detected
    actions = actions[actions["factor"] != 1.0]
    dividends = None
    if n.dividends_file and Path(n.dividends_file).exists():
        dividends = pd.read_csv(n.dividends_file)
        dividends.columns = [c.lower().strip() for c in dividends.columns]
        dividends["ex_date"] = pd.to_datetime(dividends["ex_date"]).dt.normalize()
    adjusted = adjust_prices(frames, actions, dividends)
    log_path = raw_dir.parent / "corporate_actions_detected.csv"
    actions.sort_values(["symbol", "ex_date"]).to_csv(log_path, index=False)
    membership = point_in_time_universe(frames["value"], frames["close"], int(n.universe_size),
                                        int(n.universe_lookback), int(n.min_listing_days))
    return {"frames": adjusted, "membership": membership, "actions": actions, "renames": renames,
            "dividends_included": dividends is not None, "actions_log": log_path}
