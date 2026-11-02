"""Command-line interface: ``indquant <command> [options]``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import load_config, parse_overrides, parse_value

log = logging.getLogger("indquant")


def _setup_logging(cfg, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        log_dir = Path(cfg.logging.dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "indquant.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers,
                        force=True)
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cfg(args, extra: dict | None = None):
    overrides = parse_overrides(args.set)
    if getattr(args, "source", None):
        overrides["data.source"] = args.source
    overrides.update(extra or {})
    return load_config(args.config, overrides)


def _load(cfg, args, refresh: bool = False):
    from .data import load_market_data

    return load_market_data(cfg, as_of=getattr(args, "as_of", None), refresh=refresh)


def _sleeves(args) -> tuple[str, ...]:
    return tuple(s.strip() for s in args.sleeves.split(",") if s.strip())


def _print_metrics(metrics: dict) -> None:
    keys = ["cagr", "volatility", "sharpe", "max_drawdown", "deflated_sharpe", "alpha", "alpha_tstat",
            "excess_cagr", "annual_turnover", "annual_cost_drag", "win_rate", "n_round_trips"]
    cols = [c for c in ("total", "core", "swing", "benchmark", "development", "holdout") if c in metrics]
    df = pd.DataFrame({c: {k: metrics[c].get(k) for k in keys} for c in cols})
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 140):
        print(df.to_string())


# ------------------------------------------------------------------------- commands
def cmd_fetch(args) -> int:
    cfg = _cfg(args)
    data = _load(cfg, args, refresh=True)
    print(f"Loaded {data.close.shape[1]} symbols, {len(data.dates)} sessions "
          f"({data.dates[0].date()} -> {data.dates[-1].date()}) from {data.source}.")
    missing = data.close.iloc[-1].isna().sum()
    print(f"{missing} symbols have no price on the last date.")
    return 0


def cmd_research(args) -> int:
    from .data.universe import eligibility_mask
    from .features.factors import compute_factors
    from .research.factor_analysis import factor_report, forward_returns, ic_decay
    from .research.hypotheses import check_against_hypotheses

    cfg = _cfg(args)
    data = _load(cfg, args)
    mask = eligibility_mask(data, cfg)
    factors = compute_factors(data)
    h = int(args.horizon)
    report = check_against_hypotheses(factor_report(factors, forward_returns(data, h), mask, h))
    out = Path(cfg.general.output_dir) / "research"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "factor_report.csv")
    decay = pd.DataFrame({n: ic_decay(f, data, mask) for n, f in factors.items()}).T
    decay.to_csv(out / "ic_decay.csv")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 160,
                           "display.max_colwidth", 40):
        print(report.drop(columns=["reference"]).to_string())
        print("\nMean rank IC by horizon (trading days):")
        print(decay.to_string())
    print(f"\nSaved to {out}/")
    return 0


def cmd_backtest(args) -> int:
    from .backtest.report import write_report
    from .backtest.runner import run_backtest
    from .research.trials import config_hash, data_key, distinct_trials, record_trial

    cfg = _cfg(args)
    plain = cfg.to_dict()
    n_logged = distinct_trials(cfg.logging.trial_log, data_key(plain), config_hash(plain))
    if n_logged > int(cfg.validation.n_trials):
        cfg = cfg.copy_with({"validation.n_trials": n_logged})
        log.info("Trial log: %d distinct configurations tested on this data -> n_trials=%d", n_logged, n_logged)
    data = _load(cfg, args)
    res = run_backtest(cfg, data, _sleeves(args))
    research = None
    if args.with_research:
        from .data.universe import eligibility_mask
        from .features.factors import compute_factors
        from .research.factor_analysis import factor_report, forward_returns
        from .research.hypotheses import check_against_hypotheses

        mask = eligibility_mask(data, cfg)
        research = check_against_hypotheses(factor_report(compute_factors(data), forward_returns(data, 21), mask, 21))
        research = research.drop(columns=["reference"])
    if args.null_test:
        from .research.null_tests import random_signal_test

        for name, strat in res.strategies.items():
            nt = random_signal_test(cfg, data, strat, res.equity[name], n_runs=int(args.null_test))
            res.metrics[name]["null_test_p_value"] = nt["p_value"]
            res.metrics[name]["null_mean_sharpe"] = nt["null_mean"]
            if nt["n_runs"] < 19:
                verdict = f"inconclusive - use --null-test 19 or more (min attainable p is {1 / (nt['n_runs'] + 1):.2f})"
            elif nt["p_value"] < 0.05:
                verdict = "signal adds value beyond random picks under the same rules"
            else:
                verdict = "NOT distinguishable from random picks"
            res.warnings.append(f"{name} random-signal test: Sharpe {nt['actual_sharpe']:.2f} vs random "
                                f"mean {nt['null_mean']:.2f} (95th pct {nt['null_95pct']:.2f}), "
                                f"p = {nt['p_value']:.3f} -> {verdict}.")
    out = Path(args.out or Path(cfg.general.output_dir) / "backtest" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = write_report(res, out, title=f"indquant backtest ({data.source})", research=research)
    if not args.no_log:
        record_trial(cfg.logging.trial_log, plain, res.metrics, ",".join(_sleeves(args)), args.label or "")
    _print_metrics(res.metrics)
    print("\n" + "\n".join(f"! {w}" for w in res.warnings))
    print(f"\nReport: {path}")
    return 0


def cmd_sweep(args) -> int:
    from .backtest.runner import run_backtest
    from .research.trials import record_trial

    base = _cfg(args)
    data = _load(base, args)
    values = [parse_value(v) for v in args.values.split(",")]
    n = len(values)
    rows = []
    for v in values:
        cfg = base.copy_with({args.param: v, "validation.n_trials": max(n, int(base.validation.n_trials))})
        res = run_backtest(cfg, data, _sleeves(args))
        record_trial(cfg.logging.trial_log, cfg.to_dict(), res.metrics, ",".join(_sleeves(args)), f"sweep {args.param}")
        m = res.metrics["total"]
        rows.append({args.param: v, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                     "excess_cagr": m.get("excess_cagr"), "deflated_sharpe": m["deflated_sharpe"],
                     "turnover": m.get("annual_turnover")})
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(df.to_string(index=False))
    spread = df["sharpe"].max() - df["sharpe"].min()
    print(f"\nSharpe range across values: {spread:.2f}. A robust strategy should not depend sharply on one "
          "parameter; prefer a value in a flat region over the single best one.")
    return 0


def cmd_healthcheck(args) -> int:
    from .live.healthcheck import check_data, drop_incomplete_bar, has_errors

    cfg = _cfg(args)
    data = _load(cfg, args)
    data, dropped = drop_incomplete_bar(data)
    if dropped:
        print("Note: dropped today's still-forming bar (market not closed).")
    issues = check_data(data, cfg)
    for i in issues:
        print(f"[{i.severity.upper()}] {i.message}")
    if not issues:
        print(f"OK - {data.close.shape[1]} symbols, last bar {data.dates[-1].date()}")
    return 2 if has_errors(issues) else 0


def cmd_news(args) -> int:
    from .data.sentiment import update_history
    from .data.universe import load_universe

    cfg = _cfg(args)
    new = update_history(cfg.sentiment.history_file, list(cfg.sentiment.feeds), load_universe(cfg))
    if len(new):
        with pd.option_context("display.max_colwidth", 90, "display.width", 160):
            print(new.sort_values("score").to_string(index=False))
    else:
        print("No company-matched headlines fetched.")
    return 0


def _live_prepare(cfg, args):
    from .live.healthcheck import check_data, drop_incomplete_bar, has_errors

    if cfg.sentiment.live_fetch and cfg.data.source != "synthetic":
        from .data.sentiment import update_history
        from .data.universe import load_universe

        try:
            update_history(cfg.sentiment.history_file, list(cfg.sentiment.feeds), load_universe(cfg))
        except Exception as exc:  # news is optional - never block on it
            log.warning("News fetch failed: %s", exc)
    data = _load(cfg, args)
    if cfg.data.source != "synthetic" and not getattr(args, "allow_intraday", False):
        data, dropped = drop_incomplete_bar(data)
        if dropped:
            log.info("Dropped today's incomplete bar; signals use the previous close.")
    issues = check_data(data, cfg)
    if has_errors(issues) and not args.force:
        for i in issues:
            print(f"[{i.severity.upper()}] {i.message}")
        print("Refusing to generate signals on bad data (use --force to override).")
        return None, issues
    return data, issues


def cmd_signals(args) -> int:
    from .live.paper import PaperBook
    from .live.signals import dry_run, write_signal_files

    cfg = _cfg(args)
    data, issues = _live_prepare(cfg, args)
    if data is None:
        return 2
    accounts, capital, variant = None, float(cfg.general.initial_capital), "fresh"
    db = Path(cfg.paper.db_path)
    if db.exists() and not args.fresh:
        book = PaperBook(db)
        v = book.variants()
        champ = v.loc[v["is_champion"] == 1]
        if len(champ):
            variant = champ.iloc[0]["name"]
            cfg = cfg.copy_with(champ.iloc[0]["overrides"])
            capital = float(champ.iloc[0]["capital"])
            accounts = book.load_accounts(variant) or None
    res = dry_run(cfg, data, accounts, capital, variant)
    res.health = issues
    path = write_signal_files(res, Path(cfg.general.output_dir) / "signals", "dry run (paper book not modified)")
    print(path.read_text(encoding="utf-8"))
    print(f"\nSaved: {path.parent}")
    return 0


def cmd_paper_init(args) -> int:
    from .live.paper import PaperBook

    cfg = _cfg(args)
    variant_over = parse_overrides(args.variant_set)
    cfg.copy_with(variant_over)  # validate early
    book = PaperBook(cfg.paper.db_path)
    capital = float(args.capital or cfg.general.initial_capital)
    book.add_variant(args.variant, capital, variant_over, champion=args.champion, note=args.note or "")
    print(f"Paper variant '{args.variant}' created with ₹{capital:,.0f} and overrides {variant_over or '{}'} "
          f"in {cfg.paper.db_path}. Run `indquant paper-update` after each market close.")
    return 0


def cmd_paper_update(args) -> int:
    from .live.paper import PaperBook
    from .live.signals import step_accounts, write_signal_files

    cfg = _cfg(args)
    book = PaperBook(cfg.paper.db_path)
    variants = book.variants()
    if variants.empty:
        print("No paper variants. Run `indquant paper-init` first.")
        return 1
    data, issues = _live_prepare(cfg, args)
    if data is None:
        for _, v in variants.iterrows():
            book.log_run(v["name"], None, "blocked", "; ".join(i.message for i in issues))
        return 2
    for _, v in variants.iterrows():
        if args.variant and v["name"] != args.variant:
            continue
        vcfg = cfg.copy_with(v["overrides"])
        try:
            res = step_accounts(vcfg, data, book.load_accounts(v["name"]) or None, float(v["capital"]), v["name"])
        except Exception as exc:
            book.log_run(v["name"], str(data.dates[-1].date()), "error", repr(exc))
            log.exception("Paper update failed for %s", v["name"])
            continue
        res.health = issues
        book.save(v["name"], res.accounts)
        book.log_run(v["name"], str(res.data_date.date()), "ok", f"{len(res.new_trades)} fills")
        mode = "paper trading" + (" - CHAMPION" if v["is_champion"] else " - challenger")
        path = write_signal_files(res, Path(cfg.general.output_dir) / "signals", mode)
        print(f"[{v['name']}] equity ₹{sum(res.equity.values()):,.0f}, {len(res.new_trades)} new fills -> {path}")
        if v["is_champion"]:
            print(path.read_text(encoding="utf-8"))
    return 0


def cmd_paper_report(args) -> int:
    from .backtest.metrics import trade_stats
    from .live.paper import PaperBook, compare_variants

    cfg = _cfg(args)
    book = PaperBook(cfg.paper.db_path)
    table, notes = compare_variants(book, cfg)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 140):
        print(table.to_string() if len(table) else "No variants.")
    for n in notes:
        print("- " + n)
    trades = book.trades()
    if len(trades):
        out = Path(cfg.general.output_dir) / "paper"
        out.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out / "trade_journal.csv", index=False)
        for name, t in trades.groupby("variant"):
            stats = trade_stats(t.assign(holding_days=t["holding_days"].astype(float)))
            print(f"\n[{name}] " + json.dumps(stats, default=str))
            closed = t[t["side"] == "sell"]
            if len(closed):
                by_reason = closed.groupby("reason")["pnl"].agg(["count", "sum", "mean"])
                print(by_reason.to_string())
        print(f"\nFull trade journal: {out / 'trade_journal.csv'} (review it weekly; change rules rarely).")
    return 0


def cmd_paper_promote(args) -> int:
    from .live.paper import PaperBook

    cfg = _cfg(args)
    PaperBook(cfg.paper.db_path).promote(args.variant)
    print(f"'{args.variant}' is now the champion. `indquant signals` will follow it.")
    return 0


# ------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="indquant", description="Systematic trading research for Indian equities")
    p.add_argument("--config", default=None, help="YAML config merged over config/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--source", choices=["yahoo", "csv", "synthetic"], help="shortcut for --set data.source=...")
    p.add_argument("--as-of", dest="as_of", default=None, help="truncate data at this date (YYYY-MM-DD)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download / refresh universe and prices").set_defaults(func=cmd_fetch)

    r = sub.add_parser("research", help="factor IC / quantile analysis checked against hypotheses")
    r.add_argument("--horizon", type=int, default=21)
    r.set_defaults(func=cmd_research)

    b = sub.add_parser("backtest", help="walk-forward backtest with HTML report")
    b.add_argument("--sleeves", default="core,swing")
    b.add_argument("--out", default=None)
    b.add_argument("--label", default=None, help="note stored in the trial log")
    b.add_argument("--with-research", action="store_true", help="include the factor research table")
    b.add_argument("--null-test", type=int, default=0, metavar="N", help="random-signal hypothesis test with N runs")
    b.add_argument("--no-log", action="store_true", help="do not record this run in the trial log")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("sweep", help="parameter sensitivity (each value counts as a trial)")
    s.add_argument("--param", required=True, help="dotted config key, e.g. core.top_n")
    s.add_argument("--values", required=True, help="comma separated values")
    s.add_argument("--sleeves", default="core,swing")
    s.set_defaults(func=cmd_sweep)

    sub.add_parser("healthcheck", help="data quality checks").set_defaults(func=cmd_healthcheck)
    sub.add_parser("news", help="fetch and score news headlines into the sentiment history").set_defaults(func=cmd_news)

    for name, func, helptext in (("signals", cmd_signals, "orders for the next session (dry run)"),
                                 ("paper-update", cmd_paper_update, "advance the paper book to the latest close")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--force", action="store_true", help="ignore data health errors (not recommended)")
        sp.add_argument("--allow-intraday", action="store_true", help="use today's bar even before the close")
        if name == "signals":
            sp.add_argument("--fresh", action="store_true", help="ignore the paper book and start from cash")
        else:
            sp.add_argument("--variant", default=None, help="only update this variant")
        sp.set_defaults(func=func)

    pi = sub.add_parser("paper-init", help="create a paper-trading variant")
    pi.add_argument("--variant", default="champion")
    pi.add_argument("--capital", type=float, default=None)
    pi.add_argument("--champion", action="store_true")
    pi.add_argument("--note", default=None)
    pi.add_argument("--variant-set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override stored with this variant")
    pi.set_defaults(func=cmd_paper_init)

    sub.add_parser("paper-report", help="variant league table and trade journal").set_defaults(func=cmd_paper_report)
    pp = sub.add_parser("paper-promote", help="make a variant the champion (manual decision)")
    pp.add_argument("--variant", required=True)
    pp.set_defaults(func=cmd_paper_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    _setup_logging(cfg, args.verbose)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
