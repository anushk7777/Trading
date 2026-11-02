import json
from datetime import datetime

import pandas as pd

from indquant.cli import main
from indquant.config import load_config
from indquant.data.sentiment import match_symbols, score_text, sentiment_frame
from indquant.live.healthcheck import check_data, drop_incomplete_bar, has_errors
from indquant.live.paper import PaperBook, compare_variants
from indquant.live.signals import orders_frame, step_accounts
from indquant.research.hypotheses import check_against_hypotheses
from indquant.research.trials import config_hash, distinct_trials, record_trial

from .conftest import small_overrides


def test_paper_orders_fill_next_session_and_state_round_trips(small_data, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path, **{"ml.enabled": False}))
    t0 = small_data.dates[500]
    book = PaperBook(tmp_path / "paper.db")
    book.add_variant("champ", 1_000_000, {})
    r1 = step_accounts(cfg, small_data.truncate(t0), None, 1_000_000, "champ")
    book.save("champ", r1.accounts)
    orders = orders_frame(r1)
    assert len(orders) > 0 and not r1.new_trades  # first run only creates orders
    accounts = book.load_accounts("champ")
    assert sum(len(a.pending) for a in accounts.values()) == len(orders)

    t1 = small_data.dates[501]
    r2 = step_accounts(cfg, small_data.truncate(t1), accounts, 1_000_000, "champ")
    book.save("champ", r2.accounts)
    assert r2.new_trades and all(t["date"] == t1 for t in r2.new_trades)
    filled = {t["symbol"] for t in r2.new_trades if t["side"] == "buy"}
    assert filled <= set(orders["symbol"])
    o = small_data.open.loc[t1]
    for t in r2.new_trades:
        if t["side"] == "buy":
            assert t["price"] >= o[t["symbol"]]  # buys at next open plus slippage
        else:
            assert t["reason"] in ("stop", "stop_gap", "target", "target_gap")  # same-day bracket exit
    # re-running the same day is idempotent
    r3 = step_accounts(cfg, small_data.truncate(t1), book.load_accounts("champ"), 1_000_000, "champ")
    assert not r3.new_trades
    table, _ = compare_variants(book, cfg)
    assert table.loc["champ", "days"] == 2


def test_cli_end_to_end_on_synthetic(tmp_path, capsys):
    base = []
    for k, v in small_overrides(tmp_path, **{"ml.enabled": False}).items():
        base += ["--set", f"{k}={json.dumps(v) if v is not None else 'null'}"]
    assert main(base + ["paper-init", "--variant", "champ"]) == 0
    assert main(base + ["paper-init", "--variant", "concentrated", "--variant-set", "core.top_n=10"]) == 0
    assert main(base + ["--as-of", "2017-06-01", "paper-update"]) == 0
    assert main(base + ["--as-of", "2017-06-05", "paper-update"]) == 0
    assert main(base + ["paper-report"]) == 0
    assert main(base + ["--as-of", "2017-06-05", "signals"]) == 0
    out = capsys.readouterr().out
    assert "Orders for next session" in out and "concentrated" in out
    assert main(base + ["backtest", "--sleeves", "core", "--out", str(tmp_path / "bt")]) == 0
    assert (tmp_path / "bt" / "report.html").exists()
    assert (tmp_path / "state" / "trials.csv").exists()


def test_trial_log_counts_distinct_configs(tmp_path):
    cfg = load_config().to_dict()
    log = tmp_path / "t.csv"
    m = {"total": {"sharpe": 1.0}}
    record_trial(log, cfg, m, "core")
    record_trial(log, cfg, m, "core")
    cfg2 = load_config(overrides={"core.top_n": 30}).to_dict()
    record_trial(log, cfg2, m, "core")
    key = f"{cfg['data']['source']}:{cfg['data']['universe']}:{cfg['general']['start']}"
    assert distinct_trials(log, key) == 2
    assert config_hash(cfg) != config_hash(cfg2)


def test_healthcheck_flags_stale_and_partial_bar(small_data, tmp_path):
    cfg = load_config(overrides=small_overrides(tmp_path))
    d = small_data
    d.source = "yahoo"
    try:
        issues = check_data(d, cfg, now=datetime(2030, 1, 1))
        assert has_errors(issues) and any("stale" in i.message for i in issues)
        last = d.dates[-1]
        trimmed, dropped = drop_incomplete_bar(d, now=datetime(last.year, last.month, last.day, 11, 0))
        assert dropped and trimmed.dates[-1] == d.dates[-2]
    finally:
        d.source = "synthetic"


def test_sentiment_scoring_matching_and_point_in_time(tmp_path):
    assert score_text("Infosys shares surge after profit beats estimates") > 0
    assert score_text("SEBI probe: company faces fraud investigation, stock plunges") < 0
    assert score_text("Results not weak") > 0  # negation flips
    aliases = {"INFY": ["INFY", "Infosys"], "TCS": ["TCS", "Tata Consultancy Services"]}
    assert match_symbols("Infosys wins large deal", aliases) == ["INFY"]
    assert match_symbols("Stocks to watch: TCS, Infosys", aliases) == ["INFY", "TCS"]

    dates = pd.bdate_range("2024-01-01", periods=5)
    pd.DataFrame({"symbol": ["INFY", "TCS"], "timestamp": ["2024-01-02 10:00", "2024-01-02 18:00"],
                  "score": [-1.0, 1.0]}).to_csv(tmp_path / "s.csv", index=False)
    s = sentiment_frame(tmp_path / "s.csv", dates, ["INFY", "TCS"], 3)
    assert s.loc["2024-01-01"].abs().sum() == 0
    assert s.loc["2024-01-02", "INFY"] < 0 and s.loc["2024-01-02", "TCS"] == 0  # after close -> next day
    assert s.loc["2024-01-03", "TCS"] > 0


def test_hypothesis_check_flags_unjustified_factors():
    rep = pd.DataFrame({"t_stat": [3.0, -3.0, 5.0]}, index=["mom_12_1", "low_vol", "lucky_pattern"])
    out = check_against_hypotheses(rep)
    assert out.loc["mom_12_1", "verdict"] == "supported"
    assert out.loc["low_vol", "verdict"].startswith("CONTRADICTED")
    assert out.loc["lucky_pattern", "verdict"].startswith("NO HYPOTHESIS")
