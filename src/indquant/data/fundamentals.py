"""Optional point-in-time fundamentals.

Free sources (e.g. yfinance ``.info``) only give *today's* fundamentals, which would leak
future information into a backtest. So fundamentals are only accepted from a user CSV with
the date each number became public:

    symbol,date,roe,debt_to_equity,earnings_yield,book_to_price
    RELIANCE,2019-05-15,0.12,0.75,0.045,0.38

``date`` must be the announcement / filing date, not the period end.
"""
from __future__ import annotations

import pandas as pd


def load_fundamentals(path: str, dates: pd.DatetimeIndex, symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path)
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    metrics = [c for c in raw.columns if c not in ("symbol", "date")]
    out = {}
    for metric in metrics:
        wide = raw.pivot_table(index="date", columns="symbol", values=metric, aggfunc="last")
        wide = wide.reindex(columns=symbols)
        # as-of join: value known from its publication date onwards
        union = wide.index.union(dates)
        out[metric] = wide.reindex(union).ffill().reindex(dates)
    return out
