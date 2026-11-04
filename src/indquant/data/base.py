"""Core market data container used by every other module."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class MarketData:
    """Wide OHLCV panel: each frame is indexed by trading date with one column per symbol.

    Prices must be split/dividend adjusted. NaN means the stock did not trade (not yet
    listed, suspended, or delisted). Prices are never forward-filled here.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series                       # investable benchmark close (total return if possible)
    index: pd.Series                           # regime / beta index close
    sectors: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    membership: pd.DataFrame | None = None     # bool, point-in-time index membership
    fundamentals: dict[str, pd.DataFrame] = field(default_factory=dict)
    source: str = "unknown"
    survivorship_free: bool = False
    notes: list = field(default_factory=list)   # data caveats surfaced in reports
    extra_benchmarks: dict = field(default_factory=dict)  # e.g. factor ETFs to compare against
    alt_data: dict = field(default_factory=dict)          # long tables keyed by public_date (disclosures)

    def __post_init__(self) -> None:
        idx = self.close.index
        cols = self.close.columns
        for name in FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(idx) or not frame.columns.equals(cols):
                setattr(self, name, frame.reindex(index=idx, columns=cols))
        self.benchmark = self.benchmark.reindex(idx).ffill()
        self.index = self.index.reindex(idx).ffill()
        if self.sectors is None or len(self.sectors) == 0:
            self.sectors = pd.Series("Unknown", index=cols, dtype=object)
        else:
            self.sectors = self.sectors.reindex(cols).fillna("Unknown")
        if self.membership is not None:
            self.membership = self.membership.reindex(index=idx, columns=cols).fillna(False).astype(bool)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def truncate(self, end) -> "MarketData":
        """Return data up to and including ``end`` (used to prove signals are causal)."""
        end = pd.Timestamp(end)
        sl = slice(None, end)
        return MarketData(
            open=self.open.loc[sl], high=self.high.loc[sl], low=self.low.loc[sl],
            close=self.close.loc[sl], volume=self.volume.loc[sl],
            benchmark=self.benchmark.loc[sl], index=self.index.loc[sl], sectors=self.sectors,
            membership=None if self.membership is None else self.membership.loc[sl],
            fundamentals={k: v.loc[sl] for k, v in self.fundamentals.items()},
            source=self.source, survivorship_free=self.survivorship_free, notes=list(self.notes),
            extra_benchmarks=dict(self.extra_benchmarks),
            alt_data={k: (v[v["public_date"] <= end] if isinstance(v, pd.DataFrame) and "public_date" in v else v)
                      for k, v in self.alt_data.items()},
        )

    def select(self, symbols: list[str]) -> "MarketData":
        return MarketData(
            open=self.open[symbols], high=self.high[symbols], low=self.low[symbols],
            close=self.close[symbols], volume=self.volume[symbols],
            benchmark=self.benchmark, index=self.index, sectors=self.sectors.reindex(symbols),
            membership=None if self.membership is None else self.membership[symbols],
            fundamentals={k: v.reindex(columns=symbols) for k, v in self.fundamentals.items()},
            source=self.source, survivorship_free=self.survivorship_free, notes=list(self.notes),
            extra_benchmarks=dict(self.extra_benchmarks), alt_data=dict(self.alt_data),
        )


def frames_from_long(records: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert {symbol: OHLCV frame} into {field: wide frame}."""
    out = {}
    for name in FIELDS:
        cols = {}
        for sym, df in records.items():
            col = name.capitalize()
            if col in df.columns:
                cols[sym] = df[col]
        out[name] = pd.DataFrame(cols).sort_index()
    return out


def clean_prices(frames: dict[str, pd.DataFrame], spike: float = 0.35) -> dict[str, pd.DataFrame]:
    """Basic data hygiene for free data sources.

    * non-positive prices -> NaN
    * single-day spikes that fully revert next day (bad ticks) -> NaN
    * high/low repaired so that low <= open, close <= high
    """
    close = frames["close"].where(frames["close"] > 0)
    ret = close.pct_change(fill_method=None)
    nxt = ret.shift(-1)
    bad = (ret.abs() > spike) & (nxt.abs() > spike) & (np.sign(ret) != np.sign(nxt))
    out = {}
    for name in FIELDS:
        frame = frames[name].copy()
        if name != "volume":
            frame = frame.where(frame > 0)
        frame = frame.mask(bad)
        out[name] = frame
    o, c = out["open"], out["close"]
    out["open"] = o.fillna(c)
    out["high"] = _row_max(out["high"], out["open"], c)
    out["low"] = _row_min(out["low"], out["open"], c)
    out["volume"] = out["volume"].fillna(0.0).where(c.notna())
    return out


_SPLIT_RATIOS = [r for k in (2, 3, 4, 5, 10, 20, 25, 50, 100) for r in (1 / k, float(k))]


def clean_index_series(s: pd.Series, max_move: float = 0.20, max_spike_days: int = 10) -> tuple[pd.Series, list[str]]:
    """Repair an index / index-ETF price series from a free feed.

    A broad Indian index has never moved 20% in one session (worst day: -13% on 23 Mar 2020), so any
    bigger move is bad data. A jump that returns to the old level within ``max_spike_days`` is a bad
    tick and is replaced by the last good price; a jump that never reverts is an unadjusted unit split
    and earlier prices are rescaled. Returns the series and a description of each repair."""
    x = s.dropna().astype(float)
    v = x.to_numpy().copy()
    fixes: list[str] = []
    i = 1
    while i < len(v):
        jump = v[i] / v[i - 1] - 1
        if abs(jump) <= max_move:
            i += 1
            continue
        base = v[i - 1]
        back = next((j for j in range(i + 1, min(len(v), i + max_spike_days + 1)) if abs(v[j] / base - 1) <= max_move),
                    None)
        if back is not None:
            v[i:back] = base
            fixes.append(f"{x.index[i].date()}: removed a {jump:+.0%} bad price lasting {back - i} day(s)")
            i = back
        else:
            ratio = v[i] / v[i - 1]
            clean = min(_SPLIT_RATIOS, key=lambda c: abs(np.log(ratio / c)))
            factor = clean if abs(np.log(ratio / clean)) < 0.05 else ratio   # keep that day's real move
            v[:i] *= factor
            fixes.append(f"{x.index[i].date()}: {jump:+.0%} jump that never reverted - treated as an unadjusted split")
            i += 1
    out = s.astype(float).copy()
    out.loc[x.index] = v
    return out, fixes


def _row_max(*frames: pd.DataFrame) -> pd.DataFrame:
    arr = np.fmax.reduce([f.to_numpy(dtype=float) for f in frames])
    return pd.DataFrame(arr, index=frames[0].index, columns=frames[0].columns).where(frames[-1].notna())


def _row_min(*frames: pd.DataFrame) -> pd.DataFrame:
    arr = np.fmin.reduce([f.to_numpy(dtype=float) for f in frames])
    return pd.DataFrame(arr, index=frames[0].index, columns=frames[0].columns).where(frames[-1].notna())
