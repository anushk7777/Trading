"""Public "smart money" disclosures: NSE bulk deals and SEBI insider-trading (PIT) filings.

Both are published by NSE so that ordinary investors can see them; acting on them after
publication is normal, legal investing.

Normalised long tables (one row per disclosure):
* bulk:    public_date, date, symbol, client, side (+1 buy / -1 sell), qty, price, value
* insider: public_date, symbol, person, category, mode, side, qty, value

``public_date`` is the first day the information was available to the public, which is
what every signal must key on (point-in-time). Bulk deals are published the evening of
the deal; insider filings carry a broadcast timestamp (after 17:00 counts as the next day;
if missing, the intimation date + 2 days is used conservatively).

Data arrives three ways: CSV exports from nseindia.com (most robust - column names are
matched by alias/keyword), a best-effort fetch from NSE's website, or - for tests and the
offline demo - synthetic disclosures with planted skilled and unskilled traders.
"""
from __future__ import annotations

import http.cookiejar
import json
import logging
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
BULK_COLS = ["public_date", "date", "symbol", "client", "side", "qty", "price", "value"]
INSIDER_COLS = ["public_date", "symbol", "person", "category", "mode", "side", "qty", "value"]
INSIDER_CATEGORIES = ("promoter", "director", "key managerial", "kmp")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick(df: pd.DataFrame, exact: list[str], contains: list[tuple] = (), exclude: tuple = ()) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for e in exact:
        if _norm(e) in cols:
            return cols[_norm(e)]
    for keys in contains:
        for n, c in cols.items():
            if all(k in n for k in keys) and not any(x in n for x in exclude):
                return c
    return None


def normalise_client(name: str) -> str:
    n = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    n = re.sub(r"\b(LIMITED|LTD|PVT|PRIVATE|THE|AND|CO)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _side(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    out = pd.Series(np.nan, index=series.index)
    out[s.str.startswith(("B", "ACQ", "PURCH"))] = 1.0
    out[s.str.startswith(("S", "DISP"))] = -1.0
    return out


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str.slice(0, 19), errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce", format="mixed")
    return out


# ------------------------------------------------------------------------- normalisers
def normalise_bulk(raw: pd.DataFrame) -> pd.DataFrame:
    c_date = _pick(raw, ["date", "bd_dt_date", "deal date"], [("date",)])
    c_sym = _pick(raw, ["symbol", "bd_symbol"], [("symbol",)])
    c_cli = _pick(raw, ["client name", "bd_client_name"], [("client",)])
    c_side = _pick(raw, ["buy/sell", "bd_buy_sell"], [("buy", "sell")])
    c_qty = _pick(raw, ["quantity traded", "bd_qty_trd"], [("quantity",), ("qty",)])
    c_px = _pick(raw, ["trade price / wght. avg. price", "bd_tp_watp"], [("price",)])
    if None in (c_date, c_sym, c_cli, c_side, c_qty, c_px):
        raise ValueError(f"unrecognised bulk-deal columns: {list(raw.columns)[:10]}")
    df = pd.DataFrame({
        "date": _dates(raw[c_date]).dt.normalize(), "symbol": raw[c_sym].astype(str).str.strip().str.upper(),
        "client": raw[c_cli].map(normalise_client), "side": _side(raw[c_side]),
        "qty": _num(raw[c_qty]), "price": _num(raw[c_px]),
    }).dropna(subset=["date", "side", "qty", "price"])
    df["value"] = df["qty"] * df["price"]
    df["public_date"] = df["date"]
    return df[BULK_COLS].reset_index(drop=True)


def normalise_insider(raw: pd.DataFrame) -> pd.DataFrame:
    c_sym = _pick(raw, ["symbol"], [("symbol",)])
    c_person = _pick(raw, ["name of the acquirer/disposer", "acqName"], [("acquirer",), ("nameofthe",)])
    c_cat = _pick(raw, ["category of person", "personCategory"], [("category",)])
    c_mode = _pick(raw, ["mode of acquisition", "acqMode"], [("mode",)])
    c_side = _pick(raw, ["acquisition/disposal transaction type", "tdpTransactionType"], [("transactiontype",)])
    c_qty = _pick(raw, ["no. of securities (acquired/displosed)", "secAcq"],
                  [("noofsecurities", "acquired"), ("noofsecurities", "dispos"), ("noofsecurities", "displos")],
                  exclude=("prior", "post"))
    c_val = _pick(raw, ["value of security (acquired/displosed)", "secVal"], [("value",)], exclude=("prior", "post"))
    c_bc = _pick(raw, ["broadcaste date and time", "broadcast date and time", "date"], [("broadcast",)])
    c_int = _pick(raw, ["date of initmation to company", "intimDt"], [("intimation",), ("initmation",)])
    if None in (c_sym, c_cat, c_val) or (c_bc is None and c_int is None):
        raise ValueError(f"unrecognised insider-trading columns: {list(raw.columns)[:10]}")
    bc = _dates(raw[c_bc]) if c_bc else pd.Series(pd.NaT, index=raw.index)
    late = bc.dt.hour >= 17
    public = bc.dt.normalize() + pd.to_timedelta(late.fillna(False).astype(int), unit="D")
    if c_int:
        public = public.fillna(_dates(raw[c_int]).dt.normalize() + pd.Timedelta(days=2))
    mode = raw[c_mode].astype(str) if c_mode else pd.Series("", index=raw.index)
    side = _side(raw[c_side]) if c_side else pd.Series(np.nan, index=raw.index)
    side = side.fillna(pd.Series(np.where(mode.str.lower().str.contains("purchase"), 1.0,
                                          np.where(mode.str.lower().str.contains("sale"), -1.0, np.nan)),
                                 index=raw.index))
    df = pd.DataFrame({
        "public_date": public, "symbol": raw[c_sym].astype(str).str.strip().str.upper(),
        "person": raw[c_person].map(normalise_client) if c_person else "",
        "category": raw[c_cat].astype(str).str.strip(), "mode": mode.str.strip(), "side": side,
        "qty": _num(raw[c_qty]) if c_qty else np.nan, "value": _num(raw[c_val]),
    })
    cat = df["category"].str.lower()
    keep = cat.apply(lambda x: any(k in x for k in INSIDER_CATEGORIES))
    keep &= df["mode"].str.lower().str.contains("market") & ~df["mode"].str.lower().str.contains("off")
    df = df[keep].dropna(subset=["public_date", "side", "value"])
    return df[INSIDER_COLS].reset_index(drop=True)


def _read(path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".json":
        payload = json.loads(p.read_text())
        return pd.DataFrame(payload.get("data", payload))
    return pd.read_csv(p, dtype=str)


def _merge_store(store: Path, new: pd.DataFrame, cols: list[str]) -> int:
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists():
        old = pd.read_csv(store, parse_dates=["public_date"] + (["date"] if "date" in cols else []))
        merged = pd.concat([old, new], ignore_index=True)
    else:
        old, merged = pd.DataFrame(columns=cols), new
    key = [c for c in cols if c not in ("price",)]
    merged = merged.drop_duplicates(subset=key).sort_values("public_date")
    merged.to_csv(store, index=False)
    return len(merged) - len(old)


def import_files(cfg, bulk: str | None = None, insider: str | None = None) -> dict:
    sm = cfg.smart_money
    added = {}
    if bulk:
        added["bulk"] = _merge_store(Path(sm.bulk_deals_file), normalise_bulk(_read(bulk)), BULK_COLS)
    if insider:
        added["insider"] = _merge_store(Path(sm.insider_file), normalise_insider(_read(insider)), INSIDER_COLS)
    return added


def load_disclosures(cfg) -> dict:
    sm = cfg.smart_money
    out = {}
    if sm.bulk_deals_file and Path(sm.bulk_deals_file).exists():
        out["bulk"] = pd.read_csv(sm.bulk_deals_file, parse_dates=["public_date", "date"])
    if sm.insider_file and Path(sm.insider_file).exists():
        out["insider"] = pd.read_csv(sm.insider_file, parse_dates=["public_date"])
    return out


# ------------------------------------------------------------------------- best-effort NSE fetch
def _opener():
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [("User-Agent", _UA), ("Accept", "application/json,text/plain,*/*"),
                     ("Referer", "https://www.nseindia.com/")]
    op.open("https://www.nseindia.com", timeout=20).read()  # sets session cookies
    return op


def fetch_recent(cfg, since: date | None = None, until: date | None = None, chunk_days: int = 30) -> dict:
    """Try NSE's website for disclosures between ``since`` and ``until``; store what arrives."""
    until = until or date.today()
    since = since or until - timedelta(days=30)
    added = {"bulk": 0, "insider": 0}
    try:
        op = _opener()
    except Exception as exc:
        log.warning("NSE website unavailable for disclosures (%s); import CSV exports instead.", exc)
        return added
    sm = cfg.smart_money
    d = since
    while d <= until:
        e = min(d + timedelta(days=chunk_days - 1), until)
        f, t = d.strftime("%d-%m-%Y"), e.strftime("%d-%m-%Y")
        for kind, url, normaliser, store, cols in (
            ("bulk", f"https://www.nseindia.com/api/historical/bulk-deals?from={f}&to={t}", normalise_bulk,
             sm.bulk_deals_file, BULK_COLS),
            ("insider", f"https://www.nseindia.com/api/corporates-pit?index=equities&from_date={f}&to_date={t}",
             normalise_insider, sm.insider_file, INSIDER_COLS)):
            try:
                payload = json.loads(op.open(url, timeout=30).read())
                rows = pd.DataFrame(payload.get("data", []))
                if len(rows):
                    added[kind] += _merge_store(Path(store), normaliser(rows), cols)
            except Exception as exc:  # schema changes / blocking: warn and continue
                log.warning("Disclosure fetch %s %s..%s failed: %s", kind, f, t, exc)
            time.sleep(0.5)
        d = e + timedelta(days=1)
    return added


# ------------------------------------------------------------------------- synthetic (tests / demo)
def generate_synthetic_disclosures(data, seed: int = 0, n_smart: int = 5, n_noise: int = 60,
                                   horizon: int = 63) -> dict:
    """Planted ground truth: 'smart' clients and insiders tend to buy before future out-performance."""
    rng = np.random.default_rng(seed + 101)
    close = data.close
    fwd = (close.shift(-horizon) / close - 1).sub(
        (data.benchmark.shift(-horizon) / data.benchmark - 1), axis=0)
    rank = fwd.rank(axis=1, pct=True)
    adv = (close * data.volume).rolling(63, min_periods=10).mean()
    dates, syms = close.index, np.array(close.columns)
    bulk, insider = [], []
    for i, d in enumerate(dates[:-1]):
        alive = close.iloc[i].notna().to_numpy()
        if alive.sum() < 5:
            continue
        r = rank.iloc[i].to_numpy()
        for _ in range(rng.poisson(1.5)):                       # noise clients: random picks
            j = rng.choice(np.nonzero(alive)[0])
            bulk.append((d, syms[j], f"NOISE CAPITAL {rng.integers(n_noise)}", rng.choice([1.0, -1.0]), j, i))
        if rng.random() < 0.35:                                 # smart clients: mostly top-quintile picks
            good = np.nonzero(alive & (np.nan_to_num(r) > 0.8))[0]
            pool = good if (len(good) and rng.random() < 0.7) else np.nonzero(alive)[0]
            j = rng.choice(pool)
            bulk.append((d, syms[j], f"SMART FUND {rng.integers(n_smart)}", 1.0, j, i))
        if rng.random() < 0.25:                                 # insider buys / sells
            good = np.nonzero(alive & (np.nan_to_num(r) > 0.7))[0]
            j = rng.choice(good if (len(good) and rng.random() < 0.6) else np.nonzero(alive)[0])
            side = 1.0 if rng.random() < 0.8 else -1.0
            insider.append((d, syms[j], "PROMOTER", "Promoters", "Market Purchase" if side > 0 else "Market Sale",
                            side, j, i))
    def value(j, i):
        a = adv.iat[i, j]
        return float((a if a == a else 1e7) * rng.uniform(0.005, 0.03))
    b = pd.DataFrame([(d, d, s, c, sd, np.nan, float(close.iat[i, j]), value(j, i)) for d, s, c, sd, j, i in bulk],
                     columns=BULK_COLS)
    b["qty"] = b["value"] / b["price"]
    ins = pd.DataFrame([(d + pd.Timedelta(days=1), s, p, cat, m, sd, np.nan, value(j, i))
                        for d, s, p, cat, m, sd, j, i in insider], columns=INSIDER_COLS)
    return {"bulk": b, "insider": ins}
