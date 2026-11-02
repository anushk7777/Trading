"""Real-data assembly paths (CSV source, Yahoo frame parsing, cleaning) without network access."""
import numpy as np
import pandas as pd

from indquant.config import load_config
from indquant.data import load_market_data
from indquant.data.base import clean_prices
from indquant.data.yahoo import _extract, yahoo_symbol


def _write_csv_dir(data, root):
    root.mkdir()
    for sym in data.symbols[:20]:
        df = pd.DataFrame({"Date": data.dates, "Open": data.open[sym], "High": data.high[sym], "Low": data.low[sym],
                           "Close": data.close[sym], "Volume": data.volume[sym]}).dropna()
        df.to_csv(root / f"{sym}.csv", index=False)
    for name, s in (("NIFTYBEES.NS", data.benchmark), ("^NSEI", data.index)):
        pd.DataFrame({"Date": data.dates, "Open": s, "High": s, "Low": s, "Close": s, "Volume": 1}).to_csv(
            root / f"{name}.csv", index=False)


def test_csv_source_builds_market_data(small_data, tmp_path):
    _write_csv_dir(small_data, tmp_path / "csv")
    uni = pd.DataFrame({"Symbol": small_data.symbols[:20], "Industry": small_data.sectors[:20].to_numpy(),
                        "Company Name": small_data.symbols[:20]})
    uni.to_csv(tmp_path / "uni.csv", index=False)
    cfg = load_config(overrides={"data.source": "csv", "data.csv_dir": str(tmp_path / "csv"),
                                 "data.universe": "file", "data.universe_file": str(tmp_path / "uni.csv"),
                                 "general.start": "2015-01-01"})
    d = load_market_data(cfg)
    assert d.close.shape == (len(small_data.dates), 20)
    assert d.sectors.loc[small_data.symbols[0]] == small_data.sectors.loc[small_data.symbols[0]]
    np.testing.assert_allclose(d.close.iloc[-1].to_numpy(), small_data.close.iloc[-1, :20].to_numpy())
    assert not d.survivorship_free


def test_yahoo_frame_parsing_both_layouts():
    idx = pd.date_range("2024-01-01", periods=3, tz="Asia/Kolkata")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    a = pd.DataFrame(np.arange(30, dtype=float).reshape(3, 10), index=idx,
                     columns=pd.MultiIndex.from_product([["RELIANCE.NS", "TCS.NS"], fields]))
    b = a.swaplevel(axis=1)
    for raw in (a, b):
        out = _extract(raw, "TCS.NS", single=False)
        assert list(out.columns) == fields and out.index.tz is None
        assert out["Close"].iloc[0] == a[("TCS.NS", "Close")].iloc[0]
    assert _extract(a, "INFY.NS", single=False) is None
    assert yahoo_symbol("M&M") == "M&M.NS" and yahoo_symbol("^NSEI") == "^NSEI"


def test_clean_prices_removes_reverting_spike_and_repairs_ranges():
    idx = pd.bdate_range("2024-01-01", periods=5)
    c = pd.DataFrame({"S": [100, 101, 200, 102, 103.0]}, index=idx)
    frames = {"open": c.copy(), "high": c - 5, "low": c + 5, "close": c, "volume": c * 0 + 10}
    out = clean_prices(frames)
    assert np.isnan(out["close"].loc[idx[2], "S"])
    ok = out["close"]["S"].notna()
    assert (out["high"]["S"][ok] >= out["close"]["S"][ok]).all()
    assert (out["low"]["S"][ok] <= out["close"]["S"][ok]).all()
