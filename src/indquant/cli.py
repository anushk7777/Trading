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


def cmd_disclosures(args) -> int:
    from datetime import date

    from .data.disclosures import fetch_recent, import_files

    cfg = _cfg(args)
    if args.disc_cmd == "import":
        if not (args.bulk or args.insider):
            print("Give --bulk FILE and/or --insider FILE (CSV exports from nseindia.com).")
            return 1
        added = import_files(cfg, args.bulk, args.insider)
    else:
        since = date.fromisoformat(args.since) if args.since else None
        added = fetch_recent(cfg, since=since)
    print(f"New rows stored: {added} -> {cfg.smart_money.bulk_deals_file}, {cfg.smart_money.insider_file}")
    return 0


def cmd_smart_money(args) -> int:
    from .features.smart_money import smart_money

    cfg = _cfg(args)
    data = _load(cfg, args)
    alt = data.alt_data or {}
    if alt.get("bulk") is None and alt.get("insider") is None:
        print("No disclosure data yet. Run `indquant disclosures fetch --since 2018-01-01` or import CSV exports "
              "with `indquant disclosures import --bulk bulk.csv --insider insider.csv`.")
        return 1
    with pd.option_context("display.width", 160, "display.float_format", "{:,.3f}".format):
        if alt.get("bulk") is not None and len(alt["bulk"]):
            _, tab = smart_money(data, alt["bulk"], alt["params"])
            print(f"Bulk-deal clients tested: {len(tab)}; skilled after false-discovery control "
                  f"(q <= {cfg.smart_money.fdr_q}): {int(tab['selected'].sum()) if len(tab) else 0}")
            if len(tab):
                print(tab.head(int(args.top)).rename(columns={"mean": "avg_excess_63d"}).to_string())
            recent = alt["bulk"][alt["bulk"]["public_date"] >= data.dates[-1] - pd.Timedelta(days=14)]
            skilled = set(tab.index[tab["selected"]]) if len(tab) else set()
            hits = recent[recent["client"].isin(skilled)]
            print(f"\nSkilled-client deals in the last 14 days: {len(hits)}")
            if len(hits):
                print(hits[["public_date", "symbol", "client", "side", "value"]].to_string(index=False))
        if alt.get("insider") is not None and len(alt["insider"]):
            ins = alt["insider"][alt["insider"]["public_date"] >= data.dates[-1] - pd.Timedelta(days=30)]
            buys = ins[ins["side"] > 0].groupby("symbol")["value"].sum().sort_values(ascending=False)
            print(f"\nInsider open-market net buying, last 30 days (top {int(args.top)}):")
            print(buys.head(int(args.top)).to_string() if len(buys) else "none")
    print("\nThese feed the `smart_money` and `insider_buying` factors; check `indquant research` before trusting them.")
    return 0


def cmd_data_audit(args) -> int:
    from .data.audit import compare_sources, suspicious_moves
    from .data.yahoo import download_prices

    cfg = _cfg(args)
    data = _load(cfg, args)
    out = Path(cfg.general.output_dir) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    moves = suspicious_moves(data.close, float(args.threshold))
    moves.to_csv(out / "suspicious_moves.csv", index=False)
    print(f"{len(moves)} single-day moves > {float(args.threshold):.0%} remain after adjustment "
          f"(saved to {out / 'suspicious_moves.csv'}); check them for missed splits/bonuses.")
    if data.source == "nse":
        last = data.membership.iloc[-1] if data.membership is not None else data.close.iloc[-1].notna()
        symbols = list(last[last].index[: int(args.symbols)])
        recs = download_prices(symbols, str(cfg.general.start), cfg.general.end, cfg.data.cache_dir,
                               max_age_hours=float(cfg.data.cache_max_age_hours))
        table = compare_sources(data.close, {s: r["Close"] for s, r in recs.items()})
        table.to_csv(out / "nse_vs_yahoo.csv", index=False)
        if len(table):
            bad = table[table["mismatch_days"] > 0]
            print(f"\nNSE vs Yahoo on {len(table)} stocks: median return correlation "
                  f"{table['return_corr'].median():.3f}; {len(bad)} stocks with mismatching days (>5%).")
            with pd.option_context("display.width", 160, "display.max_colwidth", 60):
                print(bad.head(20).to_string(index=False))
            print("\nFix a wrong adjustment by adding a row to nse.corporate_actions_file (factor 1 cancels one).")
        else:
            print("No overlapping Yahoo data to compare against.")
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
    from .research.expectations import build as build_expectations, save as save_expectations

    save_expectations(cfg.logging.expectations_file, build_expectations(res.equity["total"].pct_change(fill_method=None)),
                      {"config_hash": config_hash(plain), "source": data.source, "sleeves": ",".join(_sleeves(args))})
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
    rows, curves = [], {}
    for v in values:
        cfg = base.copy_with({args.param: v, "validation.n_trials": max(n, int(base.validation.n_trials))})
        res = run_backtest(cfg, data, _sleeves(args))
        record_trial(cfg.logging.trial_log, cfg.to_dict(), res.metrics, ",".join(_sleeves(args)), f"sweep {args.param}")
        m = res.metrics["total"]
        curves[str(v)] = res.equity["total"].pct_change(fill_method=None)
        rows.append({args.param: v, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                     "excess_cagr": m.get("excess_cagr"), "deflated_sharpe": m["deflated_sharpe"],
                     "turnover": m.get("annual_turnover")})
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(df.to_string(index=False))
    if len(curves) >= 2:
        from .research.overfitting import probability_of_backtest_overfitting

        try:
            pbo = probability_of_backtest_overfitting(pd.DataFrame(curves))
            print(f"\nProbability of backtest overfitting (CSCV, {pbo['n_splits']} splits): {pbo['pbo']:.0%}. "
                  f"Median out-of-sample Sharpe of the in-sample winner: {pbo['oos_sharpe_of_is_best']:.2f}.")
            print("  PBO near 0% = choosing the best value is meaningful; >= 50% = it is no better than a coin flip.")
        except ValueError as exc:
            print(f"PBO not computed: {exc}")
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
    from .research.expectations import check, load as load_expectations

    exp = load_expectations(cfg.logging.expectations_file)
    for _, v in book.variants().iterrows():
        eq = book.equity(v["name"])
        if not eq.empty:
            print(f"[{v['name']}] " + check(eq.ffill().sum(axis=1), exp)["message"])
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


# ------------------------------------------------------------------------- trade desk
def _make_desk(cfg):
    from .live.desk import DeskStore, TradeDesk, _Cache
    from .review.reviewer import RedFlagReviewer

    # fake-money desk keeps its own book so it can never mix with real trades
    store = DeskStore(cfg.desk.paper_db_path if cfg.execution.broker == "paper" else cfg.desk.db_path)
    reviewer = RedFlagReviewer(cfg, cache=_Cache(store)) if cfg.reviewer.enabled else None
    return TradeDesk(cfg, store=store, reviewer=reviewer)


def _desk_refresh(cfg, args, desk) -> dict:
    data, issues = _live_prepare(cfg, args)
    if data is None:
        raise RuntimeError("Data health check failed: " + "; ".join(i.message for i in issues))
    universe = None
    if cfg.data.source != "synthetic":
        from .data.universe import load_universe

        universe = load_universe(cfg)
    res = desk.refresh(data, universe)
    for i in issues:
        res["alerts"].append(f"[data {i.severity}] {i.message}")
    return res


def _print_plan(desk) -> None:
    st = desk.state()
    print(f"Data: {st['data_date']} · broker: {st['broker']}{' (DRY RUN)' if st['dry_run'] else ''} · "
          f"equity ₹{st['equity']:,.0f}{' · KILL SWITCH ON' if st['kill'] else ''}")
    for a in st["alerts"]:
        print(f"! {a}")
    if st["proposals"]:
        rows = []
        for p in st["proposals"]:
            stop = p["ref_price"] - p["stop_dist"] if p["stop_dist"] else None
            tgt = p["ref_price"] + p["target_dist"] if p["target_dist"] else None
            rows.append({"id": p["id"], "sleeve": p["sleeve"], "order": f"{p['side']} {p['symbol']}", "qty": p["qty"],
                         "limit": p["limit_price"], "stop": stop, "target": tgt, "risk_inr": p["risk_inr"],
                         "ai": (p["review"] or {}).get("verdict", "-"), "status": p["status"]})
        with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
            print("\nNext-session plan:\n" + pd.DataFrame(rows).to_string(index=False))
    else:
        print("\nNo orders proposed.")
    if st["positions"]:
        cols = ["sleeve", "symbol", "shares", "avg_cost", "last", "pnl", "stop", "target", "risk_to_stop"]
        with pd.option_context("display.width", 160):
            print("\nPositions:\n" + pd.DataFrame(st["positions"])[cols].to_string(index=False))


def cmd_desk(args) -> int:
    cfg = _cfg(args)
    desk = _make_desk(cfg)
    a = args.desk_cmd
    if a == "init":
        desk.init(float(args.capital or cfg.general.initial_capital))
        print(f"Desk initialised with ₹{float(args.capital or cfg.general.initial_capital):,.0f} "
              f"(broker: {desk.broker.name}). Next: `indquant console` or `indquant desk refresh`.")
        return 0
    if a == "refresh":
        res = _desk_refresh(cfg, args, desk)
        print(f"{res['proposals']} proposal(s), {res['reviewed']} AI-reviewed.")
        _print_plan(desk)
        return 0
    if a == "status":
        _print_plan(desk)
        return 0
    if a == "tca":
        try:
            data = _load(cfg, args)
        except Exception as exc:  # TCA vs signal close still works without data
            log.warning("No market data for TCA vs open: %s", exc)
            data = None
        table, summ = desk.tca(data)
        if len(table):
            cols = [c for c in ("data_date", "symbol", "side", "fill_qty", "ref_price", "next_open", "fill_price",
                                "vs_signal_close_bps", "vs_open_bps") if c in table]
            with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
                print(table[cols].to_string(index=False))
        print("\n" + summ["message"])
        return 0
    if a == "journal":
        data = _load(cfg, args)
        print(desk.settle_journal(data).to_string(index=False) or "Nothing settled yet.")
        print("\nAlpha vs benchmark after N sessions. If 'veto:*' / 'ai_veto' rows have LOWER alpha than "
              "'proposed', the filters are avoiding losers; if higher, they are blocking winners.")
        return 0
    actions = {
        "approve": lambda: desk.approve(args.id, args.override, args.qty, args.price),
        "skip": lambda: desk.skip(args.id, args.reason or "") or {"message": "skipped"},
        "fill": lambda: desk.confirm_fill(args.id, args.qty, args.price),
        "cancel": lambda: desk.cancel(args.id) or {"message": "cancelled"},
        "exit": lambda: desk.exit_position(args.sleeve, args.symbol),
        "stopped": lambda: desk.stopped_out(args.sleeve, args.symbol, args.price),
        "stop": lambda: desk.set_stop(args.sleeve, args.symbol, args.stop, args.target),
        "kill": lambda: desk.kill_switch(args.reason or "cli"),
        "resume": desk.resume,
        "flatten": lambda: desk.flatten(args.confirm or ""),
        "sync": desk.sync,
    }
    try:
        res = actions[a]()
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1
    for k, v in (res or {}).items():
        print(f"{k}: {v}")
    return 0


def cmd_console(args) -> int:
    from .live.console import ConsoleApp

    cfg = _cfg(args)
    desk = _make_desk(cfg)
    if not desk.store.accounts():
        desk.init(float(cfg.general.initial_capital))
        print(f"Desk initialised with ₹{float(cfg.general.initial_capital):,.0f}.")
    app = ConsoleApp(desk, lambda: _desk_refresh(cfg, args, desk), cfg.desk.host, int(args.port or cfg.desk.port))
    app.serve(open_browser=not args.no_browser)
    return 0


def cmd_broker_login(args) -> int:
    from .execution.brokers import kite_complete_login, kite_login_url

    cfg = _cfg(args)
    if not args.request_token:
        print("1. Open this URL, log in to Zerodha, and copy the `request_token` from the page you land on:\n")
        print("   " + kite_login_url(cfg))
        print("\n2. Run: indquant broker-login --request-token <token>")
        return 0
    kite_complete_login(cfg, args.request_token)
    print(f"Kite session saved to {cfg.execution.kite_token_file} (valid for today).")
    return 0


# ------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="indquant", description="Systematic trading research for Indian equities")
    p.add_argument("--config", default=None,
                   help="YAML config merged over config/default.yaml (default: $INDQUANT_CONFIG or config/my.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--source", choices=["yahoo", "nse", "csv", "synthetic"], help="shortcut for --set data.source=...")
    p.add_argument("--as-of", dest="as_of", default=None, help="truncate data at this date (YYYY-MM-DD)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download / refresh universe and prices").set_defaults(func=cmd_fetch)

    di = sub.add_parser("disclosures", help="bulk deals / insider trades: import NSE CSV exports or fetch")
    dsub2 = di.add_subparsers(dest="disc_cmd", required=True)
    imp = dsub2.add_parser("import")
    imp.add_argument("--bulk", default=None, help="bulk-deals CSV/JSON exported from nseindia.com")
    imp.add_argument("--insider", default=None, help="insider-trading (PIT) CSV/JSON exported from nseindia.com")
    fe = dsub2.add_parser("fetch")
    fe.add_argument("--since", default=None, help="YYYY-MM-DD (default: last 30 days)")
    di.set_defaults(func=cmd_disclosures)

    sm = sub.add_parser("smart-money", help="skilled bulk-deal clients (luck-corrected) and recent insider buying")
    sm.add_argument("--top", type=int, default=15)
    sm.set_defaults(func=cmd_smart_money)

    da = sub.add_parser("data-audit", help="cross-check NSE vs Yahoo prices and flag suspicious moves")
    da.add_argument("--symbols", type=int, default=100, help="number of liquid stocks to cross-check")
    da.add_argument("--threshold", type=float, default=0.30)
    da.set_defaults(func=cmd_data_audit)

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

    co = sub.add_parser("console", help="open the one-click trade desk in your browser")
    co.add_argument("--port", type=int, default=None)
    co.add_argument("--no-browser", action="store_true")
    co.add_argument("--force", action="store_true", help="ignore data health errors on refresh")
    co.add_argument("--allow-intraday", action="store_true")
    co.set_defaults(func=cmd_console)

    bl = sub.add_parser("broker-login", help="Zerodha Kite Connect daily login")
    bl.add_argument("--request-token", default=None)
    bl.set_defaults(func=cmd_broker_login)

    dk = sub.add_parser("desk", help="trade desk from the terminal (same actions as the console)")
    dsub = dk.add_subparsers(dest="desk_cmd", required=True)
    di = dsub.add_parser("init")
    di.add_argument("--capital", type=float, default=None)
    for name in ("refresh",):
        x = dsub.add_parser(name)
        x.add_argument("--force", action="store_true")
        x.add_argument("--allow-intraday", action="store_true")
    for name in ("status", "journal", "resume", "sync", "tca"):
        dsub.add_parser(name)
    for name in ("approve", "skip", "fill", "cancel"):
        x = dsub.add_parser(name)
        x.add_argument("id", type=int)
        if name == "approve":
            x.add_argument("--override", default=None, help="reason for overriding an AI veto")
            x.add_argument("--qty", type=int, default=None)
            x.add_argument("--price", type=float, default=None, help="limit price")
        if name == "fill":
            x.add_argument("--qty", type=int, required=True)
            x.add_argument("--price", type=float, required=True)
        if name == "skip":
            x.add_argument("--reason", default=None)
    for name in ("exit", "stopped", "stop"):
        x = dsub.add_parser(name)
        x.add_argument("sleeve")
        x.add_argument("symbol")
        if name == "stopped":
            x.add_argument("--price", type=float, required=True)
        if name == "stop":
            x.add_argument("--stop", type=float, required=True)
            x.add_argument("--target", type=float, default=None)
    k = dsub.add_parser("kill")
    k.add_argument("--reason", default=None)
    f = dsub.add_parser("flatten")
    f.add_argument("--confirm", default=None, help='must be exactly "SELL ALL"')
    dk.set_defaults(func=cmd_desk)
    pp = sub.add_parser("paper-promote", help="make a variant the champion (manual decision)")
    pp.add_argument("--variant", required=True)
    pp.set_defaults(func=cmd_paper_promote)
    return p


def resolve_config_path(explicit: str | None) -> str | None:
    """--config wins; otherwise $INDQUANT_CONFIG; otherwise config/my.yaml in the current folder if present."""
    import os

    if explicit:
        return explicit
    env = os.environ.get("INDQUANT_CONFIG")
    if env:
        return env
    local = Path("config") / "my.yaml"
    return str(local) if local.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.config = resolve_config_path(args.config)
    cfg = load_config(args.config, parse_overrides(args.set))
    _setup_logging(cfg, args.verbose)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
