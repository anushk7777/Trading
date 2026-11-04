"""Data audit: reconcile two independent price sources and flag suspicious moves.

Industry practice is never to trust one data vendor. Daily returns of the NSE-adjusted
series are compared with Yahoo's adjusted series; a day where they disagree by more than
a few percent almost always means a corporate action was applied by one source and not
the other (or applied on the wrong day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_sources(primary: pd.DataFrame, secondary: dict[str, pd.Series], tol: float = 0.05) -> pd.DataFrame:
    rows = []
    for sym, s2 in secondary.items():
        if sym not in primary.columns:
            continue
        both = pd.concat([primary[sym], s2.rename("b")], axis=1, join="inner").dropna()
        if len(both) < 20:
            continue
        r = both.pct_change(fill_method=None).dropna()
        diff = (r.iloc[:, 0] - r.iloc[:, 1]).abs()
        bad = diff[diff > tol]
        worst = diff.idxmax()
        # the source showing the bigger one-day move is usually the one with the missed adjustment / bad tick
        suspect = "ours" if abs(r.iloc[:, 0].loc[worst]) > abs(r.iloc[:, 1].loc[worst]) else "yahoo"
        rows.append({"symbol": sym, "days": len(r), "return_corr": float(r.iloc[:, 0].corr(r.iloc[:, 1])),
                     "mismatch_days": int(len(bad)), "worst_date": str(worst.date()),
                     "worst_diff": float(diff.max()), "likely_wrong": suspect if len(bad) else "",
                     "mismatch_dates": ", ".join(str(d.date()) for d in bad.index[:5])})
    df = pd.DataFrame(rows)
    return df.sort_values(["mismatch_days", "worst_diff"], ascending=False) if len(df) else df


def suspicious_moves(close: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Single-day moves larger than ``threshold`` left in the adjusted data (check each one)."""
    r = close.pct_change(fill_method=None)
    idx = np.argwhere((r.abs() > threshold).to_numpy())
    rows = [{"symbol": close.columns[j], "date": str(close.index[i].date()), "return": float(r.iat[i, j])}
            for i, j in idx]
    return pd.DataFrame(rows, columns=["symbol", "date", "return"])
