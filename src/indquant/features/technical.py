"""Technical indicators on wide (date x symbol) frames. All are causal (use data <= t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = wilder(delta.clip(lower=0), n)
    down = wilder((-delta).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(close.notna())


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    a = (high - low).to_numpy()
    b = (high - prev).abs().to_numpy()
    c = (low - prev).abs().to_numpy()
    tr = np.fmax(np.fmax(a, b), c)
    return pd.DataFrame(tr, index=close.index, columns=close.columns).where(close.notna())


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return wilder(true_range(high, low, close), n)


def macd_hist(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    return (line - ema(line, signal)) / close


def bollinger_pctb(close: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return (close - (mid - k * sd)) / (2 * k * sd).replace(0, np.nan)


def volume_z(volume: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    lv = np.log1p(volume)
    mu = lv.rolling(n, min_periods=n).mean()
    sd = lv.rolling(n, min_periods=n).std().replace(0, np.nan)
    return (lv - mu) / sd


def ret(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1


def breakout(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Close relative to the prior n-day high (> 0 means a new breakout)."""
    prior_high = close.shift(1).rolling(n, min_periods=n).max()
    return close / prior_high - 1


def technical_features(data) -> dict[str, pd.DataFrame]:
    c, h, lo, v = data.close, data.high, data.low, data.volume
    daily = c.pct_change(fill_method=None)
    atr14 = atr(h, lo, c, 14)
    idx = data.index
    idx_ret = idx.pct_change(fill_method=None)
    feats = {
        "ret_1": ret(c, 1),
        "ret_5": ret(c, 5),
        "ret_10": ret(c, 10),
        "ret_21": ret(c, 21),
        "ret_63": ret(c, 63),
        "vol_21": daily.rolling(21, min_periods=15).std(),
        "atr_pct": atr14 / c,
        "rsi_14": rsi(c, 14),
        "dist_ma20": c / sma(c, 20) - 1,
        "dist_ma50": c / sma(c, 50) - 1,
        "dist_ma200": c / sma(c, 200) - 1,
        "macd_hist": macd_hist(c),
        "bb_pctb": bollinger_pctb(c),
        "vol_z": volume_z(v),
        "breakout_20": breakout(c, 20),
        "rs_63": ret(c, 63).sub(idx / idx.shift(63) - 1, axis=0),
    }
    # market context (same value for every stock on a date)
    ctx = {
        "mkt_ret_21": idx / idx.shift(21) - 1,
        "mkt_vol_21": idx_ret.rolling(21, min_periods=15).std(),
        "mkt_dist_ma200": idx / idx.rolling(200, min_periods=200).mean() - 1,
    }
    for name, series in ctx.items():
        feats[name] = pd.DataFrame(np.repeat(series.to_numpy()[:, None], c.shape[1], axis=1),
                                   index=c.index, columns=c.columns).where(c.notna())
    return feats
