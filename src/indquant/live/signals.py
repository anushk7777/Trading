"""Live signal generation and paper-trading updates.

The same strategy objects and engine used for backtests are stepped forward over new
sessions, so live signals are exactly what the backtest would have done.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..backtest.account import Account
from ..backtest.costs import CostModel
from ..backtest.engine import Engine
from ..backtest.runner import build_strategies
from .healthcheck import Issue

log = logging.getLogger(__name__)


@dataclass
class SignalResult:
    variant: str
    data_date: pd.Timestamp
    accounts: dict
    strategies: dict
    equity: dict
    health: list[Issue] = field(default_factory=list)
    new_trades: list[dict] = field(default_factory=list)


def step_accounts(cfg, data, accounts: dict[str, Account] | None, capital: float, variant: str = "default",
                  strategies: dict | None = None) -> SignalResult:
    """Advance (or create) sleeve accounts to the last bar of ``data``.

    New accounts start at the last bar: nothing is simulated in the past, the first call
    only produces orders for the next session.
    """
    strategies = strategies or build_strategies(cfg)
    engine = Engine(data, CostModel.from_config(cfg.costs), cash_yield=float(cfg.risk.cash_yield))
    accounts = dict(accounts or {})
    last = len(data.dates) - 1
    equity, new_trades = {}, []
    for name, strat in strategies.items():
        if not hasattr(strat, "data"):
            strat.prepare(data)
        acct = accounts.get(name)
        if acct is None:
            alloc = capital * float(cfg.allocation[name])
            acct = Account(name=name, cash=alloc, peak_equity=alloc)
            acct.meta["initial_equity"] = alloc
            start = last
        else:
            start = int(data.dates.searchsorted(pd.Timestamp(acct.last_date), side="right"))
        acct.history, acct.trades = [], []
        for i in range(start, last + 1):
            engine.step(i, strat, acct)
        accounts[name] = acct
        equity[name] = acct.equity(engine.close_ff.iloc[last])
        new_trades += acct.trades
    return SignalResult(variant, data.dates[last], accounts, strategies, equity, new_trades=new_trades)


def dry_run(cfg, data, accounts: dict[str, Account] | None, capital: float, variant: str = "default") -> SignalResult:
    return step_accounts(cfg, data, copy.deepcopy(accounts) if accounts else None, capital, variant)


# ---------------------------------------------------------------------- reporting
def orders_frame(res: SignalResult) -> pd.DataFrame:
    rows = []
    for sleeve, acct in res.accounts.items():
        for o in acct.pending:
            px = o.ref_price or float("nan")
            rows.append({
                "sleeve": sleeve, "symbol": o.symbol, "side": o.side.upper(), "shares": abs(o.shares),
                "ref_close": round(px, 2), "approx_value": round(abs(o.shares) * px, 0), "reason": o.reason,
                "stop": round(px - o.stop_dist, 2) if o.stop_dist else None,
                "target": round(px + o.target_dist, 2) if o.target_dist else None,
                "valid_sessions": o.ttl,
            })
    return pd.DataFrame(rows, columns=["sleeve", "symbol", "side", "shares", "ref_close", "approx_value", "reason",
                                       "stop", "target", "valid_sessions"])


def holdings_frame(res: SignalResult) -> pd.DataFrame:
    rows = []
    for sleeve, acct in res.accounts.items():
        for sym, p in acct.positions.items():
            last = p.last_price or p.entry_price
            rows.append({"sleeve": sleeve, "symbol": sym, "shares": p.shares, "entry_date": p.entry_date,
                         "avg_cost": round(p.avg_cost, 2), "last": round(last, 2),
                         "unrealised_pct": round(last / p.avg_cost - 1, 4) if p.avg_cost else None,
                         "stop": round(p.stop, 2) if p.stop else None, "target": round(p.target, 2) if p.target else None})
    return pd.DataFrame(rows)


def write_signal_files(res: SignalResult, out_dir: str | Path, mode: str) -> Path:
    day = res.data_date.date()
    out = Path(out_dir) / str(day) / res.variant
    out.mkdir(parents=True, exist_ok=True)
    orders = orders_frame(res)
    holdings = holdings_frame(res)
    orders.to_csv(out / "orders.csv", index=False)
    holdings.to_csv(out / "holdings.csv", index=False)

    lines = [f"# indquant signals - {res.variant} - data as of close {day}", "",
             f"Mode: **{mode}**. Orders below are for the **next trading session's open** "
             "(place as AMO / market-at-open orders, or skip them).", ""]
    errors = [i for i in res.health if i.severity == "error"]
    warns = [i for i in res.health if i.severity == "warning"]
    if errors or warns:
        lines.append("## Data health")
        lines += [f"- **{i.severity.upper()}**: {i.message}" for i in errors + warns]
        lines.append("")
    total = sum(res.equity.values())
    lines.append(f"## Equity: ₹{total:,.0f}")
    lines += [f"- {k}: ₹{v:,.0f}" for k, v in res.equity.items()]
    lines.append("")

    for name, strat in res.strategies.items():
        gov = getattr(strat, "last_governor", None)
        if gov is not None and gov.reasons:
            lines.append(f"**{name} risk governor active:** " + "; ".join(gov.reasons))
            lines.append("")
    core = res.strategies.get("core")
    if core is not None and core.diagnostics:
        d = core.diagnostics[-1]
        if str(pd.Timestamp(d["date"]).date()) == str(day):
            lines.append(f"Core rebalance today ({d.get('trigger', 'schedule')}): {d['n_selected']} names, "
                         f"target exposure {d['exposure']:.0%}, predicted vol {d['pred_vol']:.1%}, "
                         f"market regime {'risk-on' if d['risk_on'] else 'RISK-OFF'}, "
                         f"ML {'active' if d['ml_active'] else 'not trained yet'}.")
            lines.append("")
        else:
            lines.append(f"Core: no rebalance due (last rebalance {res.accounts['core'].meta.get('last_rebalance')}).")
            lines.append("")

    lines.append("## Orders for next session")
    lines.append(_md_table(orders) if len(orders) else "_No orders._")
    lines.append("")
    swing = res.strategies.get("swing")
    if swing is not None and swing.last_decisions:
        lines.append("## Swing candidates considered today (why / why not)")
        lines.append(_md_table(pd.DataFrame(swing.last_decisions).head(25)))
        lines.append("")
    lines.append("## Current holdings")
    lines.append(_md_table(holdings) if len(holdings) else "_None._")
    lines.append("")
    lines.append("_Research software, not investment advice. Check every order before placing it._")
    path = out / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
