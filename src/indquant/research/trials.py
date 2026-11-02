"""Trial log: every backtest configuration you run is recorded.

Selection bias is the main reason backtests fail live: run 50 variants, keep the best,
and its Sharpe is inflated by luck. Logging every trial lets the Deflated Sharpe Ratio
use the *real* number of configurations tried on the same data, instead of trusting the
researcher to report it.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

IGNORED_SECTIONS = {"logging", "paper", "healthcheck"}
IGNORED_KEYS = {("general", "output_dir"), ("validation", "n_trials"), ("validation", "bootstrap_reps"),
                ("data", "cache_dir"), ("data", "cache_max_age_hours")}
FIELDS = ["timestamp", "config_hash", "data_key", "sleeves", "start", "end", "sharpe", "cagr", "max_drawdown",
          "excess_cagr", "label"]


def config_hash(cfg: dict) -> str:
    plain = {}
    for section, values in cfg.items():
        if section in IGNORED_SECTIONS:
            continue
        if isinstance(values, dict):
            plain[section] = {k: v for k, v in values.items() if (section, k) not in IGNORED_KEYS}
        else:
            plain[section] = values
    blob = json.dumps(plain, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def data_key(cfg: dict) -> str:
    d = cfg["data"]
    return f"{d['source']}:{d['universe']}:{cfg['general']['start']}"


def distinct_trials(path: str | Path, key: str, include_hash: str | None = None) -> int:
    hashes = set()
    p = Path(path)
    if p.exists():
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("data_key") == key:
                    hashes.add(row["config_hash"])
    if include_hash:
        hashes.add(include_hash)
    return max(len(hashes), 1)


def record_trial(path: str | Path, cfg: dict, metrics: dict, sleeves: str, label: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    m = metrics.get("total", {})
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "config_hash": config_hash(cfg),
                    "data_key": data_key(cfg), "sleeves": sleeves, "start": m.get("start"), "end": m.get("end"),
                    "sharpe": m.get("sharpe"), "cagr": m.get("cagr"), "max_drawdown": m.get("max_drawdown"),
                    "excess_cagr": m.get("excess_cagr"), "label": label})
