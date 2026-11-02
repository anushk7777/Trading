"""SQLite paper-trading book with champion / challenger variants.

Each *variant* is a set of config overrides traded on paper in parallel (e.g. the current
champion vs. a challenger with different factor weights). Variants are compared with
statistics that account for how many variants are being tried; promotion is never
automatic - the report only recommends, and only after a minimum track record.
"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..backtest.account import Account
from ..backtest.metrics import deflated_sharpe, max_drawdown, returns_from_equity, sharpe

SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    name TEXT PRIMARY KEY, overrides TEXT NOT NULL, capital REAL NOT NULL,
    created TEXT NOT NULL, is_champion INTEGER NOT NULL DEFAULT 0, note TEXT
);
CREATE TABLE IF NOT EXISTS state (
    variant TEXT NOT NULL, sleeve TEXT NOT NULL, json TEXT NOT NULL, updated TEXT NOT NULL,
    PRIMARY KEY (variant, sleeve)
);
CREATE TABLE IF NOT EXISTS equity (
    variant TEXT NOT NULL, sleeve TEXT NOT NULL, date TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL,
    exposure REAL, n_positions INTEGER, PRIMARY KEY (variant, sleeve, date)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, variant TEXT NOT NULL, sleeve TEXT NOT NULL, date TEXT NOT NULL,
    symbol TEXT NOT NULL, side TEXT NOT NULL, shares INTEGER NOT NULL, price REAL NOT NULL, value REAL NOT NULL,
    costs REAL NOT NULL, slippage REAL NOT NULL, reason TEXT, pnl REAL, holding_days INTEGER
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, variant TEXT NOT NULL, run_at TEXT NOT NULL, data_date TEXT,
    status TEXT NOT NULL, message TEXT
);
"""


class PaperBook:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------------- variants
    def add_variant(self, name: str, capital: float, overrides: dict, champion: bool = False, note: str = "") -> None:
        with self._conn() as con:
            if con.execute("SELECT 1 FROM variants WHERE name=?", (name,)).fetchone():
                raise ValueError(f"Paper variant '{name}' already exists")
            has_champion = con.execute("SELECT 1 FROM variants WHERE is_champion=1").fetchone()
            con.execute("INSERT INTO variants VALUES (?,?,?,?,?,?)",
                        (name, json.dumps(overrides), capital, datetime.now().isoformat(timespec="seconds"),
                         1 if (champion or not has_champion) else 0, note))
            if champion:
                con.execute("UPDATE variants SET is_champion = (name = ?)", (name,))

    def variants(self) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query("SELECT * FROM variants ORDER BY created", con)
        df["overrides"] = df["overrides"].map(json.loads)
        return df

    def promote(self, name: str) -> None:
        with self._conn() as con:
            if not con.execute("SELECT 1 FROM variants WHERE name=?", (name,)).fetchone():
                raise ValueError(f"No paper variant named '{name}'")
            con.execute("UPDATE variants SET is_champion = (name = ?)", (name,))

    # ---------------------------------------------------------------- state
    def load_accounts(self, variant: str) -> dict[str, Account]:
        with self._conn() as con:
            rows = con.execute("SELECT sleeve, json FROM state WHERE variant=?", (variant,)).fetchall()
        return {sleeve: Account.from_dict(json.loads(js)) for sleeve, js in rows}

    def save(self, variant: str, accounts: dict[str, Account]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as con:
            for sleeve, acct in accounts.items():
                con.execute("INSERT OR REPLACE INTO state VALUES (?,?,?,?)",
                            (variant, sleeve, json.dumps(acct.to_dict(), default=str), now))
                for h in acct.history:
                    con.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?,?)",
                                (variant, sleeve, str(pd.Timestamp(h["date"]).date()), h["equity"], h["cash"],
                                 h["exposure"], h["n_positions"]))
                for t in acct.trades:
                    con.execute("INSERT INTO trades (variant, sleeve, date, symbol, side, shares, price, value, costs, "
                                "slippage, reason, pnl, holding_days) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (variant, sleeve, str(pd.Timestamp(t["date"]).date()), t["symbol"], t["side"],
                                 int(t["shares"]), t["price"], t["value"], t["costs"], t["slippage"], t["reason"],
                                 t["pnl"], t["holding_days"]))

    def log_run(self, variant: str, data_date: str | None, status: str, message: str = "") -> None:
        with self._conn() as con:
            con.execute("INSERT INTO runs (variant, run_at, data_date, status, message) VALUES (?,?,?,?,?)",
                        (variant, datetime.now().isoformat(timespec="seconds"), data_date, status, message))

    def equity(self, variant: str) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query("SELECT * FROM equity WHERE variant=? ORDER BY date", con, params=(variant,))
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot_table(index="date", columns="sleeve", values="equity", aggfunc="last")

    def trades(self, variant: str | None = None) -> pd.DataFrame:
        with self._conn() as con:
            if variant:
                return pd.read_sql_query("SELECT * FROM trades WHERE variant=? ORDER BY date, id", con, params=(variant,))
            return pd.read_sql_query("SELECT * FROM trades ORDER BY variant, date, id", con)


def compare_variants(book: PaperBook, cfg) -> tuple[pd.DataFrame, list[str]]:
    """Variant league table plus promotion recommendations (never automatic)."""
    variants = book.variants()
    rf = float(cfg.risk.risk_free_rate)
    rows, series = [], {}
    for _, v in variants.iterrows():
        eq = book.equity(v["name"])
        if eq.empty:
            rows.append({"variant": v["name"], "champion": bool(v["is_champion"]), "days": 0})
            continue
        total = eq.ffill().sum(axis=1)
        series[v["name"]] = total
        rets = returns_from_equity(total)
        rows.append({"variant": v["name"], "champion": bool(v["is_champion"]), "days": len(total),
                     "total_return": total.iloc[-1] / float(v["capital"]) - 1,
                     "sharpe": sharpe(rets, rf) if len(rets) > 20 else math.nan,
                     "max_drawdown": max_drawdown(total)})
    table = pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()
    notes: list[str] = []
    champ = variants.loc[variants["is_champion"] == 1, "name"]
    if champ.empty or champ.iloc[0] not in series:
        return table, notes
    champ = champ.iloc[0]
    n = max(len(series), 1)
    min_days = int(cfg.paper.min_days_before_promotion)
    conf = float(cfg.paper.promotion_confidence)
    for name, total in series.items():
        if name == champ:
            continue
        joined = pd.concat([total, series[champ]], axis=1, join="inner").dropna()
        days = len(joined)
        if days < min_days:
            notes.append(f"{name}: {days}/{min_days} days of paper trading - too early to judge (keep running).")
            continue
        diff = joined.iloc[:, 0].pct_change() - joined.iloc[:, 1].pct_change()
        dsr = deflated_sharpe(diff.dropna(), n_trials=n)
        table.loc[name, "prob_beats_champion"] = dsr
        if dsr >= conf:
            notes.append(f"{name}: beats champion '{champ}' with deflated probability {dsr:.2f} over {days} days - "
                         f"candidate for promotion (run `indquant paper-promote --variant {name}` if you agree).")
        else:
            notes.append(f"{name}: no significant edge over '{champ}' (deflated probability {dsr:.2f}). Keep champion.")
    return table, notes
