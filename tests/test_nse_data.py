"""NSE bhavcopy pipeline on synthetic files in both real formats (legacy and UDiFF)."""
import io
import urllib.error
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from indquant.config import load_config
from indquant.data import load_market_data
from indquant.data.nse_bhavcopy import UDIFF_FROM, bhavcopy_url, download_range, load_nse_panel, parse_bhavcopy

LEGACY_COLS = ["SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE", "TOTTRDQTY", "TOTTRDVAL",
               "TIMESTAMP", "TOTALTRADES", "ISIN"]
UDIFF_COLS = ["TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs",
              "XpryDt", "FininstrmActlXpryDt", "StrkPric", "OptnTp", "FinInstrmNm", "OpnPric", "HghPric", "LwPric",
              "ClsPric", "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "ChngInOpnIntrst",
              "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd", "SsnId", "NewBrdLotQty", "Rmks", "Rsvd1", "Rsvd2",
              "Rsvd3", "Rsvd4"]


def _zip(csv_text: str, name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


def _file(d: date, rows: list[dict]) -> bytes:
    if d >= UDIFF_FROM:
        recs = [{c: "" for c in UDIFF_COLS} | {
            "TradDt": d.isoformat(), "BizDt": d.isoformat(), "Sgmt": "CM", "Src": "NSE", "FinInstrmTp": "STK",
            "ISIN": r["isin"], "TckrSymb": r["sym"], "SctySrs": r.get("series", "EQ"), "OpnPric": r["o"],
            "HghPric": r["h"], "LwPric": r["l"], "ClsPric": r["c"], "LastPric": r["c"], "PrvsClsgPric": r["pc"],
            "TtlTradgVol": r["v"], "TtlTrfVal": r["v"] * r["c"]} for r in rows]
        return _zip(pd.DataFrame(recs, columns=UDIFF_COLS).to_csv(index=False), "bhav.csv")
    recs = [{"SYMBOL": r["sym"], "SERIES": r.get("series", "EQ"), "OPEN": r["o"], "HIGH": r["h"], "LOW": r["l"],
             "CLOSE": r["c"], "LAST": r["c"], "PREVCLOSE": r["pc"], "TOTTRDQTY": r["v"], "TOTTRDVAL": r["v"] * r["c"],
             "TIMESTAMP": d.strftime("%d-%b-%Y").upper(), "TOTALTRADES": 100, "ISIN": r["isin"]} for r in rows]
    return _zip(pd.DataFrame(recs, columns=LEGACY_COLS).to_csv(index=False), "cm.csv")


START, END = date(2024, 6, 3), date(2024, 8, 30)
HOLIDAY = date(2024, 6, 17)
BONUS_DAY, SPLIT_DAY, DELIST_AFTER, RENAME_DAY = date(2024, 7, 1), date(2024, 7, 15), date(2024, 7, 10), date(2024, 7, 22)


def _market() -> dict[str, bytes]:
    rng = np.random.default_rng(3)
    px = {"AAA": 100.0, "BBB": 400.0, "CCC": 1000.0, "DDD": 50.0, "EEE": 200.0}
    files, prev = {}, dict(px)
    d = START
    while d <= END:
        if d.weekday() < 5 and d != HOLIDAY:
            rows = []
            for s in list(px):
                if s == "DDD" and d > DELIST_AFTER:
                    continue
                pc = prev[s]
                vol = 10_000
                if s == "BBB" and d == BONUS_DAY:      # 1:1 bonus, exchange adjusts prev close
                    px[s] /= 2
                    pc = prev[s] / 2
                    vol *= 2
                if s == "CCC" and d == SPLIT_DAY:      # 1:5 split, prev close NOT adjusted
                    px[s] /= 5
                    vol *= 5
                px[s] *= float(np.exp(rng.normal(0, 0.01)))
                c = round(px[s], 2)
                sym = "FFF" if (s == "EEE" and d >= RENAME_DAY) else s
                rows.append({"sym": sym, "isin": f"INE{s}01", "o": round(c * 0.999, 2), "h": round(c * 1.01, 2),
                             "l": round(c * 0.99, 2), "c": c, "pc": round(pc, 2), "v": vol})
                prev[s] = c
            rows.append({"sym": "BOND1", "isin": "INB01", "series": "N1", "o": 100, "h": 100, "l": 100, "c": 100,
                         "pc": 100, "v": 1})
            files[bhavcopy_url(d)] = _file(d, rows)
        d += timedelta(days=1)
    return files


@pytest.fixture(scope="module")
def market_files():
    return _market()


def _fetch(files):
    def fetch(url):
        if url not in files:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return files[url]
    return fetch


def _cfg(tmp_path, **extra):
    o = {"data.source": "nse", "general.start": str(START), "general.end": str(END),
         "nse.raw_dir": str(tmp_path / "nse" / "raw"), "nse.universe_size": 3, "nse.universe_lookback": 5,
         "nse.min_listing_days": 5, "nse.request_pause": 0, "data.cache_dir": str(tmp_path / "cache")}
    o.update(extra)
    return load_config(overrides=o)


def test_both_formats_parse_to_one_schema(market_files):
    legacy = parse_bhavcopy(market_files[bhavcopy_url(date(2024, 6, 3))])
    udiff = parse_bhavcopy(market_files[bhavcopy_url(date(2024, 7, 8))])
    assert list(legacy.columns) == list(udiff.columns)
    assert "BOND1" not in set(legacy["symbol"]) | set(udiff["symbol"])
    assert legacy["date"].iloc[0] == pd.Timestamp("2024-06-03")
    assert udiff["date"].iloc[0] == pd.Timestamp("2024-07-08")
    assert legacy.loc[legacy.symbol == "AAA", "value"].iloc[0] > 0


def test_download_caches_and_remembers_holidays(tmp_path, market_files):
    stats = download_range(START, END, tmp_path / "raw", fetch=_fetch(market_files), pause=0)
    assert stats["holidays"] == 1 and stats["downloaded"] == len(market_files)
    again = download_range(START, END, tmp_path / "raw", fetch=_fetch({}), pause=0)
    assert again["downloaded"] == 0 and again["cached"] == len(market_files)


def test_panel_adjusts_actions_stitches_renames_keeps_delisted(tmp_path, market_files, monkeypatch):
    cfg = _cfg(tmp_path)
    panel = load_nse_panel(cfg, refresh=True, fetch=_fetch(market_files))
    close = panel["frames"]["close"]
    acts = panel["actions"].set_index("symbol")
    assert acts.loc["BBB", "source"] == "exchange_prev_close" and acts.loc["BBB", "factor"] == pytest.approx(0.5, abs=0.01)
    assert acts.loc["CCC", "source"] == "gap_heuristic" and acts.loc["CCC", "factor"] == pytest.approx(0.2)
    rets = close.pct_change(fill_method=None).abs()
    assert rets[["BBB", "CCC"]].max().max() < 0.06            # continuous after adjustment
    assert "EEE" not in close.columns and close["FFF"].first_valid_index() == pd.Timestamp(START)
    assert close["DDD"].last_valid_index() == pd.Timestamp(DELIST_AFTER)
    assert panel["renames"] == [("EEE", "FFF", str(RENAME_DAY))]
    member = panel["membership"]
    assert member.sum(axis=1).max() <= 3 and member.any(axis=0).sum() >= 3
    assert not member.loc[pd.Timestamp("2024-08-30"), "DDD"]
    assert panel["actions_log"].exists()


def test_user_override_cancels_a_detection(tmp_path, market_files):
    ca = tmp_path / "ca.csv"
    pd.DataFrame({"symbol": ["CCC"], "ex_date": [str(SPLIT_DAY)], "factor": [1.0]}).to_csv(ca, index=False)
    panel = load_nse_panel(_cfg(tmp_path, **{"nse.corporate_actions_file": str(ca)}), refresh=True,
                           fetch=_fetch(market_files))
    assert "CCC" not in set(panel["actions"]["symbol"])
    assert panel["frames"]["close"]["CCC"].pct_change(fill_method=None).abs().max() > 0.7


def test_load_market_data_nse_is_survivorship_free(tmp_path, market_files, monkeypatch):
    import indquant.data.nse_bhavcopy as nb

    monkeypatch.setattr(nb, "_http_get", _fetch(market_files))
    monkeypatch.setattr("indquant.data.yahoo.download_prices", lambda *a, **k: {})
    monkeypatch.setattr("indquant.data.universe.fetch_index_constituents",
                        lambda *a, **k: pd.DataFrame({"Symbol": ["AAA"], "Industry": ["IT"]}))
    cfg = _cfg(tmp_path)
    nb.download_range(START, END, cfg.nse.raw_dir, fetch=_fetch(market_files), pause=0)
    data = load_market_data(cfg)
    assert data.survivorship_free and data.membership is not None and data.source == "nse"
    assert data.sectors["AAA"] == "IT" and data.sectors["DDD"] == "Unknown"
    assert any("equal-weight index" in n for n in data.notes) and any("dividends" in n for n in data.notes)


def test_audit_flags_missed_adjustment():
    from indquant.data.audit import compare_sources, suspicious_moves

    idx = pd.bdate_range("2024-01-01", periods=60)
    a = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 60)), index=idx)
    b = a.copy()
    b.iloc[:30] *= 2  # second source missed a 1:1 bonus
    t = compare_sources(pd.DataFrame({"X": a}), {"X": b})
    assert t.iloc[0]["mismatch_days"] == 1 and t.iloc[0]["worst_date"] == str(idx[30].date())
    assert list(suspicious_moves(pd.DataFrame({"X": b}))["date"]) == [str(idx[30].date())]
