"""Self-contained HTML backtest report (charts embedded as base64 PNG)."""
from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import drawdown_series, monthly_returns

PCT_KEYS = {"factor_alpha", "total_return", "cagr", "volatility", "max_drawdown", "alpha", "benchmark_cagr", "benchmark_max_drawdown",
            "excess_cagr", "annual_turnover", "annual_cost_drag", "win_rate", "post_tax_cagr_approx"}
ROW_ORDER = ["start", "end", "final_equity", "total_return", "cagr", "volatility", "sharpe", "sharpe_ci_low",
             "sharpe_ci_high", "sortino", "max_drawdown", "calmar", "psr", "deflated_sharpe", "alpha", "alpha_tstat",
             "beta", "information_ratio", "excess_cagr", "benchmark_cagr", "annual_turnover", "annual_cost_drag",
             "n_round_trips", "win_rate", "profit_factor", "avg_holding_days", "estimated_tax", "post_tax_cagr_approx",
             "factor_alpha", "factor_alpha_tstat"]
LABELS = {"psr": "Prob. Sharpe > 0 (PSR)", "deflated_sharpe": "Deflated Sharpe (prob.)", "alpha": "Alpha (ann.)",
          "alpha_tstat": "Alpha t-stat", "excess_cagr": "CAGR vs benchmark", "annual_turnover": "Turnover / yr",
          "annual_cost_drag": "Cost + slippage drag / yr", "n_round_trips": "Closing trades",
          "post_tax_cagr_approx": "Post-tax CAGR (approx.)", "estimated_tax": "Estimated tax (INR)",
          "factor_alpha": "Alpha after factors (ann.)", "factor_alpha_tstat": "Factor-adjusted alpha t-stat"}


def _fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if isinstance(v, str):
        return html.escape(v)
    if key in PCT_KEYS:
        return f"{v:.1%}"
    if key in ("final_equity", "estimated_tax"):
        return f"₹{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:.2f}"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(res) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = res.equity
    out = {}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(eq.index, eq["total"], label="Strategy (total)", color="#1f6feb", lw=1.8)
    ax.plot(res.benchmark.index, res.benchmark, label="Benchmark", color="#888888", lw=1.4)
    for name in [c for c in eq.columns if c not in ("total",)]:
        ax.plot(eq.index, eq[name] / eq[name].iloc[0] * eq["total"].iloc[0], lw=0.9, alpha=0.7,
                label=f"{name} (rescaled)")
    ax.set_yscale("log")
    ax.set_title("Equity curve (log scale, INR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    out["equity"] = _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(eq.index, drawdown_series(eq["total"]) * 100, 0, color="#d1242f", alpha=0.4, label="Strategy")
    ax.plot(res.benchmark.index, drawdown_series(res.benchmark) * 100, color="#888888", lw=1, label="Benchmark")
    ax.set_title("Drawdown (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    out["drawdown"] = _fig_to_b64(fig)

    rets = eq["total"].pct_change(fill_method=None)
    b = res.benchmark.pct_change(fill_method=None)
    roll = rets.rolling(252).mean() / rets.rolling(252).std() * np.sqrt(252)
    roll_b = b.rolling(252).mean() / b.rolling(252).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(roll.index, roll, color="#1f6feb", lw=1.2, label="Strategy")
    ax.plot(roll_b.index, roll_b, color="#888888", lw=1, label="Benchmark")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Rolling 1-year Sharpe (raw returns)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    out["rolling"] = _fig_to_b64(fig)

    if not res.exposure.empty:
        fig, ax = plt.subplots(figsize=(10, 2.4))
        for c in res.exposure.columns:
            ax.plot(res.exposure.index, res.exposure[c] * 100, lw=1, label=c)
        ax.set_title("Invested exposure per sleeve (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        out["exposure"] = _fig_to_b64(fig)
    return out


def _metrics_table(metrics: dict) -> str:
    cols = [c for c in ("total", "core", "swing", "benchmark", "development", "holdout") if c in metrics]
    cols += [c for c in metrics if c.startswith("bm:")]
    head = "".join(f"<th>{html.escape(c[3:] if c.startswith('bm:') else c.title())}</th>" for c in cols)
    rows = []
    for key in ROW_ORDER:
        if not any(key in metrics[c] for c in cols):
            continue
        cells = "".join(f"<td>{_fmt(key, metrics[c].get(key))}</td>" for c in cols)
        rows.append(f"<tr><th>{html.escape(LABELS.get(key, key.replace('_', ' ').title()))}</th>{cells}</tr>")
    return f"<table class='num'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _monthly_table(equity: pd.Series) -> str:
    t = monthly_returns(equity)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"]
    head = "".join(f"<th>{m}</th>" for m in months)
    body = []
    for year, row in t.iterrows():
        cells = []
        for col in list(range(1, 13)) + ["Year"]:
            v = row.get(col, np.nan)
            if pd.isna(v):
                cells.append("<td></td>")
            else:
                cls = "pos" if v > 0 else "neg"
                cells.append(f"<td class='{cls}'>{v:.1%}</td>")
        body.append(f"<tr><th>{year}</th>{''.join(cells)}</tr>")
    return f"<table class='num small'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _df_table(df: pd.DataFrame, pct_cols=(), float_fmt="{:.3f}") -> str:
    if df is None or df.empty:
        return "<p class='muted'>None.</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"<td>{'' if pd.isna(v) else (f'{v:.1%}' if c in pct_cols else float_fmt.format(v))}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append(f"<tr><th>{html.escape(str(idx))}</th>{''.join(cells)}</tr>")
    return f"<table class='num'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--card:#f6f8fa;--pos:#1a7f37;--neg:#cf222e;--warn:#9a6700;--warnbg:#fff8c5}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--card:#161b22;--pos:#3fb950;--neg:#f85149;--warn:#d29922;--warnbg:#2b2111}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px 16px;}
main{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)} .warn{background:var(--warnbg);border-left:4px solid var(--warn);padding:8px 12px;margin:6px 0;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child{text-align:left} thead th{background:var(--card)}
.small td,.small th{font-size:12px;padding:3px 5px} td.pos{color:var(--pos)} td.neg{color:var(--neg)}
.scroll{overflow-x:auto} img{max-width:100%;height:auto;background:#fff;border-radius:4px}
"""


def write_report(res, out_dir: str | Path, title: str = "indquant backtest", research: pd.DataFrame | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts = _charts(res)

    res.equity.to_csv(out / "equity.csv")
    trades = res.trades.drop(columns=["tax_lots"], errors="ignore")
    trades.to_csv(out / "trades.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(res.metrics, fh, indent=2, default=str)

    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p class='muted'>{res.equity.index[0].date()} → {res.equity.index[-1].date()} · "
             f"initial capital ₹{res.cfg['general']['initial_capital']:,.0f} · "
             f"allocation core {res.cfg['allocation']['core']:.0%} / swing {res.cfg['allocation']['swing']:.0%} · "
             f"benchmark {html.escape(str(res.cfg['data']['benchmark']))}</p>"]
    parts.append("<h2>Read this first</h2>")
    parts += [f"<div class='warn'>{html.escape(w)}</div>" for w in res.warnings]
    parts.append("<h2>Performance summary</h2><div class='scroll'>" + _metrics_table(res.metrics) + "</div>")
    parts.append("<p class='muted'>All returns are net of statutory charges, brokerage and modelled slippage. "
                 "Development / holdout split the total portfolio at the start of the final "
                 f"{res.cfg['validation']['holdout_years']} years.</p>")
    for key, cap in (("equity", ""), ("drawdown", ""), ("rolling", ""), ("exposure", "")):
        if key in charts:
            parts.append(f"<img alt='{key} chart' src='data:image/png;base64,{charts[key]}'>")
    parts.append("<h2>Monthly returns (total)</h2><div class='scroll'>" + _monthly_table(res.equity["total"]) + "</div>")

    att = getattr(res, "attribution", None) or {}
    if att:
        tbl = att["table"].copy()
        tbl.index = [{"alpha": "Alpha (annualised)", "MKT": "Market beta", "MOM": "Momentum",
                      "LOWVOL": "Low volatility", "SMB": "Size (small minus big)", "VALUE": "Value",
                      "QUALITY": "Quality"}.get(i, i) for i in tbl.index]
        parts.append("<h2>Factor attribution</h2><p class='muted'>Daily returns regressed on the market and on "
                     "long-short factor portfolios from the same universe. Significant loadings mean part of the "
                     f"return is a known factor premium. R² {att['r2']:.2f}, {att['n_obs']} days.</p>"
                     + _df_table(tbl, float_fmt="{:.3f}"))
    st = getattr(res, "stress", None) or {}
    if st and len(st.get("historical", [])):
        parts.append("<h2>Stress test: historical Indian market crises</h2>"
                     + _df_table(st["historical"].set_index("event"),
                                 pct_cols=("strategy", "benchmark", "strategy_max_dd")))
    if st and len(st.get("hypothetical", [])):
        parts.append("<h2>Stress test: shocks to the final portfolio</h2>"
                     + _df_table(st["hypothetical"].set_index("scenario"), pct_cols=("portfolio_return",),
                                 float_fmt="{:,.0f}"))
    if research is not None and not research.empty:
        parts.append("<h2>Factor research (rank IC, non-overlapping monthly samples)</h2><div class='scroll'>"
                     + _df_table(research, pct_cols=("pct_positive", "q_spread_ann", "top_q_ann")) + "</div>")

    for name, strat in res.strategies.items():
        model = getattr(strat, "ml_model", None)
        if model is not None and model.importance is not None:
            imp = model.importance.mean().sort_values(ascending=False).head(15)
            imp = (imp / imp.sum()).to_frame("share of splits")
            parts.append(f"<h2>{html.escape(name.title())} ML feature importance (avg over refits)</h2>"
                         + _df_table(imp, pct_cols=("share of splits",)))

    for name, acct in res.accounts.items():
        last = res.equity.index[-1]
        rows = [{"symbol": s, "shares": p.shares, "entry": p.entry_date, "avg_cost": p.avg_cost,
                 "last": p.last_price, "stop": p.stop, "target": p.target} for s, p in acct.positions.items()]
        df = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
        parts.append(f"<h2>{html.escape(name.title())} holdings at {last.date()}</h2><div class='scroll'>"
                     + _df_table(df, float_fmt="{:,.2f}") + "</div>")
        sells = res.trades[(res.trades["sleeve"] == name) & (res.trades["side"] == "sell")] if len(res.trades) else []
        if len(sells):
            reasons = sells["reason"].value_counts().to_frame("count")
            parts.append("<p class='muted'>Exit reasons: " + ", ".join(f"{k}: {v}" for k, v in reasons["count"].items()) + "</p>")

    if len(res.taxes):
        parts.append("<h2>Estimated capital-gains tax by financial year</h2>"
                     + _df_table(res.taxes.to_frame("tax (INR)"), float_fmt="{:,.0f}")
                     + "<p class='muted'>Approximation: FIFO lots, STCG/LTCG rates from config, no loss carry-forward.</p>")

    parts.append("<h2>Disclaimer</h2><p class='muted'>Research software, not investment advice. The author is not a "
                 "SEBI-registered investment adviser. Past (and simulated) performance does not guarantee future "
                 "results. You can lose money.</p>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style>"
           f"</head><body><main>{''.join(parts)}</main></body></html>")
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
