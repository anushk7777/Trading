"""Local CSV / parquet source: one file per symbol with Date,Open,High,Low,Close,Volume.

Use this for paid or self-built datasets (e.g. adjusted NSE bhavcopy history), which are
the only way to get survivorship-free data including delisted companies.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp") or df.columns[0]
    df = df.rename(columns={cols[k]: k.capitalize() for k in ("open", "high", "low", "close", "volume") if k in cols})
    adj_key = cols.get("adj close") or cols.get("adj_close")
    if adj_key and "Close" in df.columns:
        factor = df[adj_key] / df["Close"]
        for c in ("Open", "High", "Low", "Close"):
            if c in df.columns:
                df[c] = df[c] * factor
    df.index = pd.to_datetime(df[date_col])
    df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_csv_dir(directory: str | Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    files = {p.stem: p for p in list(directory.glob("*.csv")) + list(directory.glob("*.parquet"))}
    wanted = symbols if symbols is not None else sorted(files)
    out = {}
    for sym in wanted:
        path = files.get(sym) or files.get(sym.replace(".NS", ""))
        if path is None:
            log.warning("No CSV for %s in %s", sym, directory)
            continue
        out[sym] = _read(path)
    return out
