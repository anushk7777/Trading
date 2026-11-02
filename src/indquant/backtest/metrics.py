"""Performance and statistical-significance metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change(fill_method=None).dropna()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if len(equity) else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(rets: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    ex = rets - (1 + rf) ** (1 / periods) + 1
    sd = ex.std()
    return float(ex.mean() / sd * math.sqrt(periods)) if sd and sd > 0 else float("nan")


def sortino(rets: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    ex = rets - (1 + rf) ** (1 / periods) + 1
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * math.sqrt(periods)) if downside > 0 else float("nan")


def alpha_beta(rets: pd.Series, bench: pd.Series, rf: float = 0.0, periods: int = 252) -> tuple[float, float, float]:
    """Jensen alpha (annualised), beta and alpha t-statistic from daily OLS."""
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    if len(df) < 30:
        return float("nan"), float("nan"), float("nan")
    d_rf = (1 + rf) ** (1 / periods) - 1
    y = df.iloc[:, 0].to_numpy() - d_rf
    x = df.iloc[:, 1].to_numpy() - d_rf
    res = stats.linregress(x, y)
    se = res.intercept_stderr if hasattr(res, "intercept_stderr") else float("nan")
    t = res.intercept / se if se and se > 0 else float("nan")
    return float(res.intercept * periods), float(res.slope), float(t)


def information_ratio(rets: pd.Series, bench: pd.Series, periods: int = 252) -> float:
    active = (rets - bench).dropna()
    sd = active.std()
    return float(active.mean() / sd * math.sqrt(periods)) if sd and sd > 0 else float("nan")


def probabilistic_sharpe(rets: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012): probability that the true (per-period) Sharpe
    exceeds ``sr_benchmark``, accounting for sample length, skew and fat tails."""
    r = rets.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe(rets: pd.Series, n_trials: int = 1, sr_variance: float | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Adjusts the PSR benchmark for selection bias when ``n_trials`` configurations were
    tried. ``sr_variance`` is the variance of per-period Sharpe ratios across trials;
    if unknown it defaults to the sampling variance under the null (1/T).
    """
    r = rets.dropna()
    n = len(r)
    if n < 30:
        return float("nan")
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0)
    var = sr_variance if sr_variance is not None else 1.0 / n
    sr0 = math.sqrt(var) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr0)


def bootstrap_sharpe_ci(rets: pd.Series, rf: float = 0.0, reps: int = 1000, block: int = 20,
                        seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Stationary block bootstrap confidence interval for the annualised Sharpe ratio."""
    r = rets.dropna().to_numpy()
    n = len(r)
    if n < 60:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d_rf = (1 + rf) ** (1 / 252) - 1
    out = np.empty(reps)
    for k in range(reps):
        idx = np.empty(n, dtype=int)
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            length = min(rng.geometric(1 / block), n - pos)
            idx[pos:pos + length] = (start + np.arange(length)) % n
            pos += length
        s = r[idx] - d_rf
        sd = s.std()
        out[k] = s.mean() / sd * math.sqrt(252) if sd > 0 else 0.0
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def trade_stats(trades: pd.DataFrame) -> dict:
    sells = trades[(trades["side"] == "sell") & trades["pnl"].notna()] if len(trades) else trades
    if len(sells) == 0:
        return {"n_round_trips": 0}
    pnl = sells["pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "n_round_trips": int(len(sells)),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_holding_days": float(sells["holding_days"].astype(float).mean()),
        "exit_reasons": sells["reason"].value_counts().to_dict(),
    }


def estimate_tax(trades: pd.DataFrame, stcg_rate: float, ltcg_rate: float, ltcg_exemption: float) -> pd.Series:
    """Approximate Indian capital-gains tax per financial year (Apr-Mar) from FIFO lots.

    Simplifications: STCL offsets STCG then LTCG; LTCL offsets LTCG only; no loss carry
    forward; no grandfathering; exemption applied per sleeve set as given.
    """
    if len(trades) == 0:
        return pd.Series(dtype=float)
    rows = []
    for _, t in trades[trades["side"] == "sell"].iterrows():
        d = pd.Timestamp(t["date"])
        fy = d.year if d.month >= 4 else d.year - 1
        for lot in t["tax_lots"] or []:
            rows.append({"fy": fy, "gain": lot["gain"], "long": lot["days"] > 365})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    taxes = {}
    for fy, g in df.groupby("fy"):
        st = g.loc[~g["long"], "gain"].sum()
        lt = g.loc[g["long"], "gain"].sum()
        if st < 0:
            lt += st
            st = 0.0
        tax = stcg_rate * max(st, 0.0) + ltcg_rate * max(lt - ltcg_exemption, 0.0)
        taxes[f"FY{fy}-{str(fy + 1)[-2:]}"] = tax
    return pd.Series(taxes, name="tax")


def summarize(equity: pd.Series, bench: pd.Series | None, rf: float, trades: pd.DataFrame | None = None,
              n_trials: int = 1, bootstrap_reps: int = 0) -> dict:
    equity = equity.dropna()
    rets = returns_from_equity(equity)
    out = {
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else float("nan"),
        "cagr": cagr(equity),
        "volatility": float(rets.std() * math.sqrt(252)) if len(rets) > 1 else float("nan"),
        "sharpe": sharpe(rets, rf),
        "sortino": sortino(rets, rf),
        "max_drawdown": max_drawdown(equity),
    }
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else float("nan")
    excess = rets - ((1 + rf) ** (1 / 252) - 1)
    out["psr"] = probabilistic_sharpe(excess)
    out["deflated_sharpe"] = deflated_sharpe(excess, n_trials)
    if bench is not None and len(bench.dropna()) > 1:
        b = bench.reindex(equity.index).ffill()
        b_rets = returns_from_equity(b)
        a, beta, t = alpha_beta(rets, b_rets, rf)
        out.update({"alpha": a, "beta": beta, "alpha_tstat": t,
                    "information_ratio": information_ratio(rets, b_rets.reindex(rets.index)),
                    "benchmark_cagr": cagr(b), "benchmark_sharpe": sharpe(b_rets, rf),
                    "benchmark_max_drawdown": max_drawdown(b),
                    "excess_cagr": out["cagr"] - cagr(b)})
    if bootstrap_reps:
        lo, hi = bootstrap_sharpe_ci(rets, rf, reps=bootstrap_reps)
        out["sharpe_ci_low"], out["sharpe_ci_high"] = lo, hi
    if trades is not None and len(trades):
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        tr = trades[(pd.to_datetime(trades["date"]) >= equity.index[0]) & (pd.to_datetime(trades["date"]) <= equity.index[-1])]
        avg_eq = float(equity.mean())
        out["annual_turnover"] = float(tr["value"].sum() / 2 / avg_eq / years)
        out["annual_cost_drag"] = float((tr["costs"].sum() + tr["slippage"].sum()) / avg_eq / years)
        out.update(trade_stats(tr))
    return out


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change(fill_method=None)
    first = equity.resample("ME").last().iloc[:1] / equity.iloc[0] - 1
    m.iloc[:1] = first.to_numpy()
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    table = df.pivot(index="year", columns="month", values="ret")
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change(fill_method=None)
    yr.iloc[:1] = (yearly.iloc[:1] / equity.iloc[0] - 1).to_numpy()
    table["Year"] = yr.to_numpy()
    return table
