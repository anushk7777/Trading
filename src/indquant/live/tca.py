"""Transaction-cost analysis (TCA): are real fills as cheap as the backtest assumed?

Implementation shortfall per fill, in basis points, measured against
* the next session's OPEN - the price the backtest fills at, so this is the true
  execution cost to compare with the model's slippage assumption; and
* the signal-day CLOSE - total cost of acting the next day (includes the overnight move).
Positive numbers are costs (paid more on buys / received less on sells).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shortfall_table(fills: pd.DataFrame, data=None) -> pd.DataFrame:
    """``fills``: desk proposals with status 'filled' (data_date, symbol, side, fill_price, ref_price, qty)."""
    if fills.empty:
        return fills
    df = fills.copy()
    sign = np.where(df["side"] == "BUY", 1.0, -1.0)
    df["vs_signal_close_bps"] = sign * (df["fill_price"] / df["ref_price"] - 1) * 1e4
    if data is not None:
        opens = []
        for _, r in df.iterrows():
            k = int(data.dates.searchsorted(pd.Timestamp(r["data_date"]), side="right"))
            o = data.open[r["symbol"]].iloc[k] if (k < len(data.dates) and r["symbol"] in data.open.columns) else np.nan
            opens.append(o)
        df["next_open"] = opens
        df["vs_open_bps"] = sign * (df["fill_price"] / df["next_open"] - 1) * 1e4
    return df


def summary(table: pd.DataFrame, model_bps: float, min_fills: int = 20) -> dict:
    if table.empty:
        return {"n": 0, "message": "No fills yet."}
    col = "vs_open_bps" if "vs_open_bps" in table and table["vs_open_bps"].notna().any() else "vs_signal_close_bps"
    s = table[col].dropna()
    out = {"n": int(len(s)), "basis": col, "mean_bps": float(s.mean()), "median_bps": float(s.median()),
           "p90_bps": float(s.quantile(0.9)), "model_bps": float(model_bps)}
    if len(s) < min_fills:
        out["message"] = (f"{len(s)} fills: median cost {out['median_bps']:.1f} bps vs model {model_bps:.1f} bps "
                          f"(need {min_fills}+ fills before recalibrating).")
    elif col == "vs_open_bps" and out["median_bps"] > model_bps * 1.5:
        out["message"] = (f"Real execution costs ({out['median_bps']:.1f} bps median) are well above the backtest "
                          f"assumption ({model_bps:.1f} bps). Set costs.slippage_bps={max(out['median_bps'], 1):.0f} and "
                          "re-run the backtest - the edge may be smaller than reported.")
    else:
        out["message"] = f"Execution costs ({out['median_bps']:.1f} bps median) are in line with the model."
    return out
