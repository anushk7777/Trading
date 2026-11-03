"""Smart-money factors from public disclosures (bulk deals, insider trades).

* insider_buying - net open-market buying by promoters / directors / KMP over the last
  ``insider_lookback`` sessions (by publication date), scaled by the stock's traded value.
* smart_money - net buying in the last ``deal_lookback`` sessions by bulk-deal clients whose
  past picks beat the market by more than luck allows.

"More than luck" is decided the way academic studies of fund-manager skill do it
(Barras, Scaillet & Wermers 2010): every client with enough fully-scored deals is tested,
and the Benjamini-Hochberg procedure keeps the false-discovery rate below ``fdr_q``. Among
thousands of clients many look brilliant by chance; this keeps only the few whose record is
unlikely to be chance. Scores are recomputed each month using only deals whose
``skill_horizon``-day outcome was already known at that date - no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def benjamini_hochberg(p: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected mask, q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    qvals_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    qvals = np.empty(m)
    qvals[order] = qvals_sorted
    return qvals <= q, qvals


def _sessions(dates: pd.DatetimeIndex, when: pd.Series) -> np.ndarray:
    return dates.searchsorted(pd.to_datetime(when).to_numpy(), side="left")


def insider_buying(data, ins: pd.DataFrame, lookback: int) -> pd.DataFrame | None:
    if ins is None or ins.empty:
        return None
    dates, cols = data.dates, {s: j for j, s in enumerate(data.symbols)}
    k = _sessions(dates, ins["public_date"])
    j = ins["symbol"].map(cols)
    ok = (k < len(dates)) & j.notna().to_numpy()
    daily = np.zeros(data.close.shape)
    np.add.at(daily, (k[ok], j[ok].astype(int).to_numpy()), (ins["side"] * ins["value"]).to_numpy()[ok])
    net = pd.DataFrame(daily, index=dates, columns=data.symbols).rolling(lookback, min_periods=1).sum()
    adv = (data.close * data.volume).rolling(126, min_periods=20).mean()
    return net / adv


def _deal_outcomes(data, bulk: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per deal: session, symbol column, and the signed excess return over ``horizon`` days after it."""
    dates, cols = data.dates, {s: j for j, s in enumerate(data.symbols)}
    df = bulk.copy()
    df["k"] = _sessions(dates, df["public_date"])
    df["j"] = df["symbol"].map(cols)
    df = df[(df["k"] < len(dates)) & df["j"].notna()].copy()
    df["j"] = df["j"].astype(int)
    entry, exit_ = df["k"].to_numpy() + 1, df["k"].to_numpy() + 1 + horizon
    T = len(dates)
    o, c = data.open.to_numpy(dtype=float), data.close.to_numpy(dtype=float)
    b = data.benchmark.to_numpy(dtype=float)
    valid = exit_ < T
    ret = np.full(len(df), np.nan)
    idx = np.nonzero(valid)[0]
    jj = df["j"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[idx] = (c[exit_[idx], jj[idx]] / o[entry[idx], jj[idx]] - 1) - (b[exit_[idx]] / b[entry[idx]] - 1)
    df["signed_excess"] = df["side"].to_numpy() * ret
    df["known_at"] = np.where(valid, exit_, np.iinfo(np.int64).max)
    return df


def client_skill(outcomes: pd.DataFrame, as_of: int, min_deals: int, fdr_q: float, shrink_n: float) -> pd.DataFrame:
    """Skill table using only deals whose outcome was known at session ``as_of``."""
    known = outcomes[(outcomes["known_at"] <= as_of) & outcomes["signed_excess"].notna()]
    if known.empty:
        return pd.DataFrame(columns=["n", "mean", "t", "p", "q", "selected", "weight"])
    g = known.groupby("client")["signed_excess"]
    tab = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(ddof=1)})
    tab = tab[tab["n"] >= min_deals].copy()
    if tab.empty:
        return pd.DataFrame(columns=["n", "mean", "t", "p", "q", "selected", "weight"])
    tab["t"] = tab["mean"] / (tab["sd"].replace(0, np.nan) / np.sqrt(tab["n"]))
    tab["p"] = 1 - norm.cdf(tab["t"].fillna(0))
    sel, qv = benjamini_hochberg(tab["p"].to_numpy(), fdr_q)
    tab["q"], tab["selected"] = qv, sel & (tab["mean"] > 0)
    tab["weight"] = np.where(tab["selected"], tab["mean"] * tab["n"] / (tab["n"] + shrink_n), 0.0)
    return tab.drop(columns="sd").sort_values("q")


def smart_money(data, bulk: pd.DataFrame, params: dict) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    if bulk is None or bulk.empty:
        return None, pd.DataFrame()
    horizon = int(params["skill_horizon"])
    out = _deal_outcomes(data, bulk, horizon)
    dates = data.dates
    months = dates.year * 12 + dates.month
    evals = [i for i in range(len(dates) - 1) if months[i] != months[i + 1]] + [len(dates) - 1]
    weights = np.zeros(len(out))
    last_table = pd.DataFrame()
    deal_k = out["k"].to_numpy()
    for a, e in enumerate(evals):
        nxt = evals[a + 1] if a + 1 < len(evals) else len(dates)
        tab = client_skill(out, e, int(params["min_deals"]), float(params["fdr_q"]), float(params["shrink_n"]))
        last_table = tab
        in_period = (deal_k > e) & (deal_k <= nxt)            # deals published after this evaluation
        if tab.empty or not in_period.any():
            continue
        w = out.loc[in_period, "client"].map(tab["weight"]).fillna(0.0).to_numpy()
        weights[in_period] = w
    daily = np.zeros(data.close.shape)
    np.add.at(daily, (deal_k, out["j"].to_numpy()), weights * out["side"].to_numpy() * out["value"].to_numpy())
    net = pd.DataFrame(daily, index=dates, columns=data.symbols).rolling(int(params["deal_lookback"]),
                                                                         min_periods=1).sum()
    adv = (data.close * data.volume).rolling(126, min_periods=20).mean()
    return net / adv, last_table


def smart_money_factors(data) -> dict[str, pd.DataFrame]:
    alt = getattr(data, "alt_data", None) or {}
    params = alt.get("params")
    if not params:
        return {}
    out = {}
    ins = insider_buying(data, alt.get("insider"), int(params["insider_lookback"]))
    if ins is not None:
        out["insider_buying"] = ins
    sm, _ = smart_money(data, alt.get("bulk"), params)
    if sm is not None:
        out["smart_money"] = sm
    return out
