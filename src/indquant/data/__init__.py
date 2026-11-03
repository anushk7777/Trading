"""Data loading entry point."""
from __future__ import annotations

import logging

import pandas as pd

from .base import MarketData, clean_prices, frames_from_long
from .synthetic import generate_synthetic_market

log = logging.getLogger(__name__)

__all__ = ["MarketData", "load_market_data", "generate_synthetic_market"]


def load_market_data(cfg, as_of=None, refresh: bool = False) -> MarketData:
    """Build a :class:`MarketData` panel according to ``cfg.data``."""
    src = cfg.data.source
    if src == "synthetic":
        s = cfg.data.synthetic
        data = generate_synthetic_market(
            n_stocks=int(s.n_stocks), n_days=int(s.n_days), start=str(s.start),
            seed=int(cfg.general.seed), alpha_strength=float(s.alpha_strength),
        )
    elif src == "nse":
        data = _load_nse(cfg, refresh=refresh)
    else:
        data = _load_real(cfg, refresh=refresh)
    if src != "synthetic" and cfg.data.extra_benchmarks:
        data.extra_benchmarks = _extra_benchmarks(cfg, data.dates, refresh)
    if cfg.smart_money.enabled:
        data.alt_data = _disclosures(cfg, data, refresh)
    if as_of is not None:
        data = data.truncate(as_of)
    if cfg.general.end:
        data = data.truncate(cfg.general.end)
    return data


def _load_real(cfg, refresh: bool) -> MarketData:
    from .universe import load_membership, load_universe

    universe = load_universe(cfg)
    symbols = universe["Symbol"].tolist()
    sectors = pd.Series(universe["Industry"].to_numpy(), index=universe["Symbol"], dtype=object)
    start, end = str(cfg.general.start), cfg.general.end
    bench_syms = list(dict.fromkeys([cfg.data.benchmark, cfg.data.regime_index]))

    if cfg.data.source == "yahoo":
        from .yahoo import download_prices

        records = download_prices(symbols + bench_syms, start, end, cfg.data.cache_dir,
                                  max_age_hours=float(cfg.data.cache_max_age_hours), refresh=refresh)
    else:
        from .csv_source import load_csv_dir

        records = load_csv_dir(cfg.data.csv_dir, symbols + bench_syms)

    missing_bench = [b for b in bench_syms if b not in records]
    if missing_bench:
        raise RuntimeError(f"Benchmark/index data missing for {missing_bench}; cannot build trading calendar")
    bench = records.pop(cfg.data.benchmark)["Close"]
    index = records[cfg.data.regime_index]["Close"] if cfg.data.regime_index in records else bench
    records.pop(cfg.data.regime_index, None)
    if not records:
        raise RuntimeError("No stock data loaded")

    frames = frames_from_long(records)
    calendar = bench.dropna().index
    frames = {k: v.reindex(calendar) for k, v in frames.items()}
    frames = clean_prices(frames)

    membership = None
    survivorship_free = False
    if cfg.data.membership_file:
        membership = load_membership(cfg.data.membership_file, calendar, list(frames["close"].columns))
        survivorship_free = True

    fundamentals = {}
    if cfg.data.fundamentals_file:
        from .fundamentals import load_fundamentals

        fundamentals = load_fundamentals(cfg.data.fundamentals_file, calendar, list(frames["close"].columns))

    log.info("Loaded %d symbols x %d days (%s to %s)", frames["close"].shape[1], len(calendar),
             calendar[0].date(), calendar[-1].date())
    return MarketData(
        open=frames["open"], high=frames["high"], low=frames["low"], close=frames["close"],
        volume=frames["volume"], benchmark=bench, index=index, sectors=sectors,
        membership=membership, fundamentals=fundamentals, source=cfg.data.source,
        survivorship_free=survivorship_free,
    )


def _load_nse(cfg, refresh: bool) -> MarketData:
    """Survivorship-free panel from NSE bhavcopies (see data/nse_bhavcopy.py)."""
    from .nse_bhavcopy import load_nse_panel

    panel = load_nse_panel(cfg, refresh=refresh)
    fr = panel["frames"]
    frames = clean_prices({k: fr[k] for k in ("open", "high", "low", "close", "volume")})
    calendar = frames["close"].index
    notes = [f"Survivorship-free NSE bhavcopy data: {frames['close'].shape[1]} symbols ever listed, "
             f"{len(panel['actions'])} split/bonus adjustments (review {panel['actions_log']}), "
             f"{len(panel['renames'])} symbol renames stitched."]
    if not panel["dividends_included"]:
        notes.append("Bhavcopy prices exclude dividends (no nse.dividends_file): strategy returns are understated "
                     "by roughly the market dividend yield (~1-1.5%/yr) relative to a total-return benchmark.")

    start, end = str(cfg.general.start), cfg.general.end
    bench_syms = list(dict.fromkeys([cfg.data.benchmark, cfg.data.regime_index]))
    recs = {}
    try:
        from .yahoo import download_prices

        recs = download_prices(bench_syms, start, end, cfg.data.cache_dir,
                               max_age_hours=float(cfg.data.cache_max_age_hours), refresh=refresh)
    except Exception as exc:  # benchmark is optional for loading; fall back below
        log.warning("Benchmark download failed: %s", exc)
    member = panel["membership"]
    ew = frames["close"].pct_change(fill_method=None).where(member).mean(axis=1).fillna(0.0)
    ew_index = (1 + ew).cumprod() * 1000
    bench = recs[cfg.data.benchmark]["Close"] if cfg.data.benchmark in recs else None
    index = recs[cfg.data.regime_index]["Close"] if cfg.data.regime_index in recs else None
    if bench is None or index is None:
        notes.append("Benchmark/index unavailable from Yahoo - using an equal-weight index of the point-in-time "
                     "universe instead (price-only).")
    bench = ew_index if bench is None else bench.reindex(calendar).ffill()
    index = ew_index if index is None else index.reindex(calendar).ffill()

    sectors = _nse_sectors(cfg, list(frames["close"].columns))
    log.info("NSE panel: %d symbols x %d days (%s to %s)", frames["close"].shape[1], len(calendar),
             calendar[0].date(), calendar[-1].date())
    return MarketData(open=frames["open"], high=frames["high"], low=frames["low"], close=frames["close"],
                      volume=frames["volume"], benchmark=bench, index=index, sectors=sectors, membership=member,
                      source="nse", survivorship_free=True, notes=notes)


def _nse_sectors(cfg, symbols: list[str]) -> pd.Series:
    from pathlib import Path

    if cfg.nse.sector_file and Path(cfg.nse.sector_file).exists():
        df = pd.read_csv(cfg.nse.sector_file)
        df.columns = [c.lower().strip() for c in df.columns]
        return pd.Series(df["sector"].to_numpy(), index=df["symbol"].astype(str)).reindex(symbols).fillna("Unknown")
    try:
        from .universe import fetch_index_constituents

        u = fetch_index_constituents("nifty500", cfg.data.cache_dir)
        return pd.Series(u["Industry"].to_numpy(), index=u["Symbol"]).reindex(symbols).fillna("Unknown")
    except Exception:
        return pd.Series("Unknown", index=symbols, dtype=object)


def _extra_benchmarks(cfg, calendar, refresh: bool) -> dict:
    try:
        from .yahoo import download_prices

        recs = download_prices(list(cfg.data.extra_benchmarks), str(cfg.general.start), cfg.general.end,
                               cfg.data.cache_dir, max_age_hours=float(cfg.data.cache_max_age_hours), refresh=refresh)
    except Exception as exc:
        log.warning("Extra benchmarks unavailable: %s", exc)
        return {}
    return {k: v["Close"].reindex(calendar).ffill() for k, v in recs.items()}


def _disclosures(cfg, data, refresh: bool) -> dict:
    """Public smart-money disclosures (bulk deals, insider trades) keyed by publication date."""
    from .disclosures import fetch_recent, generate_synthetic_disclosures, load_disclosures

    sm = cfg.smart_money
    if data.source == "synthetic":
        alt = generate_synthetic_disclosures(data, seed=int(cfg.general.seed)) if cfg.data.synthetic.disclosures else {}
    else:
        if sm.auto_fetch:
            _maybe_fetch_disclosures(cfg, fetch_recent, refresh)
        alt = load_disclosures(cfg)
    if alt:
        alt["params"] = {k: sm[k] for k in ("insider_lookback", "deal_lookback", "skill_horizon", "min_deals",
                                             "fdr_q", "shrink_n")}
    return alt


def _maybe_fetch_disclosures(cfg, fetch_recent, refresh: bool) -> None:
    """Fetch only the missing recent days, at most once every 6 hours (NSE may block automated access)."""
    import time
    from datetime import date, timedelta
    from pathlib import Path

    marker = Path(cfg.smart_money.bulk_deals_file).parent / ".last_fetch"
    if not refresh and marker.exists() and time.time() - marker.stat().st_mtime < 6 * 3600:
        return
    since = date.today() - timedelta(days=30)
    store = Path(cfg.smart_money.bulk_deals_file)
    if store.exists() and not refresh:
        try:
            newest = pd.read_csv(store, usecols=["public_date"], parse_dates=["public_date"])["public_date"].max()
            if pd.notna(newest):
                since = max(since, newest.date() + timedelta(days=1))
        except Exception:
            pass
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    if since <= date.today():
        try:
            fetch_recent(cfg, since=since)
        except Exception as exc:  # optional data: never block loading prices
            log.warning("Disclosure fetch failed: %s", exc)
