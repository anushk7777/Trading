"""Stress tests: historical Indian market crises and hypothetical shocks on today's book."""
from __future__ import annotations

import numpy as np
import pandas as pd

CRISES = [
    ("Global financial crisis", "2008-01-08", "2009-03-09"),
    ("Taper tantrum", "2013-05-20", "2013-08-28"),
    ("Demonetisation", "2016-11-08", "2016-12-26"),
    ("IL&FS / mid-cap crash", "2018-08-29", "2018-10-26"),
    ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    ("2022 rate-hike sell-off", "2022-01-17", "2022-06-17"),
]


def historical(equity: pd.Series, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for name, a, b in CRISES:
        e = equity.loc[a:b]
        if len(e) < 5 or e.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10):
            continue
        bm = bench.reindex(e.index).ffill()
        rows.append({"event": name, "from": a, "to": b, "strategy": float(e.iloc[-1] / e.iloc[0] - 1),
                     "benchmark": float(bm.iloc[-1] / bm.iloc[0] - 1),
                     "strategy_max_dd": float((e / e.cummax() - 1).min())})
    return pd.DataFrame(rows)


def hypothetical(weights: pd.Series, betas: pd.Series, sectors: pd.Series, total_equity: float) -> pd.DataFrame:
    """Loss on the current book under simple shocks (beta-scaled market moves, one-sector crash)."""
    if weights.empty:
        return pd.DataFrame()
    b = betas.reindex(weights.index).fillna(1.0).clip(0, 3)
    rows = []
    for shock in (-0.10, -0.20, -0.30):
        loss = float((weights * b * shock).sum())
        rows.append({"scenario": f"Market {shock:.0%}", "portfolio_return": loss, "inr": loss * total_equity})
    sec = sectors.reindex(weights.index).fillna("Unknown")
    by_sector = weights.groupby(sec).sum().sort_values(ascending=False)
    if len(by_sector):
        top = by_sector.index[0]
        loss = float(by_sector.iloc[0] * -0.15)
        rows.append({"scenario": f"Largest sector ({top}) -15%", "portfolio_return": loss, "inr": loss * total_equity})
    top_name = weights.idxmax()
    rows.append({"scenario": f"Largest holding ({top_name}) -50%", "portfolio_return": float(weights.max() * -0.5),
                 "inr": float(weights.max() * -0.5 * total_equity)})
    return pd.DataFrame(rows)


def run(res, data) -> dict:
    from ..features.factors import rolling_beta

    eq = res.equity["total"]
    hist = historical(eq, res.benchmark)
    close = data.close.ffill().iloc[-1]
    values = {}
    for acct in res.accounts.values():
        for s, p in acct.positions.items():
            values[s] = values.get(s, 0.0) + p.shares * float(close.get(s, np.nan))
    total = float(eq.iloc[-1])
    w = pd.Series(values, dtype=float).dropna() / total if values else pd.Series(dtype=float)
    beta = rolling_beta(data.close.pct_change(fill_method=None), data.index.pct_change(fill_method=None)).iloc[-1]
    return {"historical": hist, "hypothetical": hypothetical(w, beta, data.sectors, total)}
