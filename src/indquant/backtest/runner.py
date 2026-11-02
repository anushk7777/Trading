"""High-level backtest orchestration for one or both sleeves."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .account import Account
from .costs import CostModel
from .engine import Engine, history_frame, trades_frame
from .metrics import estimate_tax, summarize

log = logging.getLogger(__name__)
WARMUP_BARS = 260


def build_strategies(cfg, sleeves: tuple[str, ...] = ("core", "swing")) -> dict:
    from ..strategies.factor_portfolio import FactorPortfolioStrategy
    from ..strategies.swing import SwingStrategy

    out = {}
    if "core" in sleeves and cfg.core.enabled and cfg.allocation.core > 0:
        out["core"] = FactorPortfolioStrategy(cfg)
    if "swing" in sleeves and cfg.swing.enabled and cfg.allocation.swing > 0:
        out["swing"] = SwingStrategy(cfg)
    if not out:
        raise ValueError("No sleeve enabled - check core.enabled / swing.enabled / allocation")
    return out


def start_index(data, cfg) -> int:
    if cfg.general.backtest_start:
        idx = int(data.dates.searchsorted(pd.Timestamp(cfg.general.backtest_start)))
    else:
        idx = WARMUP_BARS
    if idx >= len(data.dates) - 1:
        raise ValueError("Not enough data after the warm-up period for a backtest")
    return idx


@dataclass
class BacktestResult:
    cfg: dict
    start_date: pd.Timestamp
    equity: pd.DataFrame                 # one column per sleeve + 'total'
    benchmark: pd.Series                 # scaled to initial capital
    trades: pd.DataFrame
    metrics: dict
    accounts: dict
    strategies: dict
    taxes: pd.Series
    exposure: pd.DataFrame
    warnings: list = field(default_factory=list)


def run_backtest(cfg, data, sleeves: tuple[str, ...] = ("core", "swing"), strategies: dict | None = None) -> BacktestResult:
    strategies = strategies or build_strategies(cfg, sleeves)
    cost_model = CostModel.from_config(cfg.costs)
    engine = Engine(data, cost_model, cash_yield=float(cfg.risk.cash_yield))
    i0 = start_index(data, cfg)
    capital = float(cfg.general.initial_capital)

    accounts, curves, exposure = {}, {}, {}
    for name, strat in strategies.items():
        log.info("Preparing %s sleeve (features%s)...", name, " + walk-forward ML" if cfg.ml.enabled else "")
        strat.prepare(data)
        alloc = float(cfg.allocation[name])
        acct = Account(name=name, cash=capital * alloc, peak_equity=capital * alloc)
        log.info("Simulating %s sleeve from %s", name, data.dates[i0].date())
        engine.run(strat, acct, i0)
        accounts[name] = acct
        hist = history_frame(acct)
        curves[name] = hist["equity"]
        exposure[name] = hist["exposure"]

    eq = pd.DataFrame(curves)
    leftover = capital * (1 - sum(float(cfg.allocation[n]) for n in strategies))
    days = np.arange(len(eq))
    eq["cash_reserve"] = leftover * (1 + engine.daily_yield) ** (days + 1)
    eq["total"] = eq.sum(axis=1)
    eq = eq.drop(columns="cash_reserve") if leftover == 0 else eq

    bench = data.benchmark.reindex(eq.index).ffill()
    bench_scaled = bench / bench.iloc[0] * capital
    trades = trades_frame(list(accounts.values()))

    rf = float(cfg.risk.risk_free_rate)
    n_trials = int(cfg.validation.n_trials)
    reps = int(cfg.validation.bootstrap_reps)
    metrics = {"total": summarize(eq["total"], bench_scaled, rf, trades, n_trials, reps)}
    for name in strategies:
        metrics[name] = summarize(eq[name], bench_scaled, rf, trades[trades["sleeve"] == name] if len(trades) else None,
                                  n_trials)
    metrics["benchmark"] = summarize(bench_scaled, None, rf)

    hy = float(cfg.validation.holdout_years)
    if hy > 0:
        cut = eq.index[-1] - pd.DateOffset(years=int(hy), months=int(round((hy % 1) * 12)))
        dev, hold = eq.loc[:cut, "total"], eq.loc[cut:, "total"]
        if len(dev) > 60 and len(hold) > 60:
            metrics["development"] = summarize(dev, bench_scaled.loc[:cut], rf, None, n_trials)
            metrics["holdout"] = summarize(hold, bench_scaled.loc[cut:], rf, None, n_trials)

    t = cfg.tax
    taxes = estimate_tax(trades, float(t.stcg_rate), float(t.ltcg_rate), float(t.ltcg_exemption))
    if len(taxes):
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        post = eq["total"].iloc[-1] - taxes.sum()
        metrics["total"]["estimated_tax"] = float(taxes.sum())
        metrics["total"]["post_tax_cagr_approx"] = float((post / eq["total"].iloc[0]) ** (1 / years) - 1)

    result = BacktestResult(cfg=cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg), start_date=eq.index[0],
                            equity=eq, benchmark=bench_scaled, trades=trades, metrics=metrics, accounts=accounts,
                            strategies=strategies, taxes=taxes, exposure=pd.DataFrame(exposure))
    result.warnings = build_warnings(cfg, data, result)
    return result


def build_warnings(cfg, data, res: BacktestResult) -> list[str]:
    w = []
    m = res.metrics["total"]
    if data.source == "synthetic":
        w.append("SYNTHETIC DATA: these results are from a simulated market and say nothing about real NSE returns.")
    elif not data.survivorship_free:
        w.append("SURVIVORSHIP BIAS: the universe is today's index constituents. Stocks that were dropped or "
                 "delisted in the past are missing, so historical returns are OVERSTATED. Supply "
                 "data.membership_file (point-in-time constituents) to fix this.")
    for name, strat in res.strategies.items():
        model = getattr(strat, "ml_model", None)
        if cfg.ml.enabled:
            if model is None or not model.fit_dates:
                w.append(f"{name}: ML model never trained (insufficient history) - rule/factor scores only.")
            else:
                w.append(f"{name}: ML predictions are walk-forward out-of-sample from {model.fit_dates[0].date()} "
                         f"(retrained {len(model.fit_dates)} times). Before that date the {name} sleeve used "
                         f"{'factor' if name == 'core' else 'rule-based'} scores.")
    dsr = m.get("deflated_sharpe")
    if dsr is not None and not np.isnan(dsr) and dsr < 0.95:
        w.append(f"NOT STATISTICALLY SIGNIFICANT: deflated Sharpe probability {dsr:.2f} < 0.95 "
                 f"(n_trials={cfg.validation.n_trials}). The result is consistent with luck.")
    if m.get("excess_cagr") is not None and m["excess_cagr"] < 0:
        w.append(f"UNDERPERFORMS BENCHMARK: CAGR is {m['excess_cagr']:.1%} per year below "
                 f"{cfg.data.benchmark} after costs. A low-cost index fund would have done better.")
    if "holdout" in res.metrics and res.metrics["holdout"]["sharpe"] < 0.5 * res.metrics["development"]["sharpe"]:
        w.append("HOLDOUT DEGRADATION: Sharpe in the final holdout period is less than half of the development "
                 "period - a classic sign of overfitting or a decaying edge.")
    if int(cfg.validation.n_trials) <= 1:
        w.append("Set validation.n_trials to the number of parameter sets you have tried; every extra trial "
                 "raises the bar for significance (Deflated Sharpe).")
    return w
