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
    else:
        data = _load_real(cfg, refresh=refresh)
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
