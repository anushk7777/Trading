"""Smart-money disclosures: parsing, point-in-time rules, luck-corrected skill selection."""
import numpy as np
import pandas as pd
import pytest

from indquant.config import load_config
from indquant.data.base import MarketData
from indquant.data.disclosures import (generate_synthetic_disclosures, import_files, normalise_bulk,
                                       normalise_insider)
from indquant.features.smart_money import benjamini_hochberg, client_skill, smart_money, smart_money_factors

PARAMS = {"insider_lookback": 63, "deal_lookback": 21, "skill_horizon": 63, "min_deals": 8, "fdr_q": 0.10,
          "shrink_n": 10}


def test_bulk_parsing_both_header_styles():
    csv = pd.DataFrame({"Date": ["03-NOV-2026"], "Symbol": ["TCS"], "Security Name": ["Tata Consultancy"],
                        "Client Name": ["Acme Capital Pvt. Ltd."], "Buy/Sell": ["BUY"],
                        "Quantity Traded": ["1,20,000"], "Trade Price / Wght. Avg. Price": ["4,012.50"]})
    js = pd.DataFrame({"BD_DT_DATE": ["2026-11-03"], "BD_SYMBOL": ["TCS"], "BD_CLIENT_NAME": ["ACME CAPITAL"],
                       "BD_BUY_SELL": ["SELL"], "BD_QTY_TRD": [5000], "BD_TP_WATP": [4000.0]})
    a, b = normalise_bulk(csv), normalise_bulk(js)
    assert list(a.columns) == list(b.columns)
    assert a.loc[0, "client"] == b.loc[0, "client"] == "ACME CAPITAL"
    assert a.loc[0, "side"] == 1 and b.loc[0, "side"] == -1
    assert a.loc[0, "value"] == pytest.approx(120000 * 4012.5)
    assert a.loc[0, "public_date"] == pd.Timestamp("2026-11-03")


def test_insider_parsing_filters_and_publication_time():
    raw = pd.DataFrame({
        "SYMBOL": ["AAA", "AAA", "BBB", "CCC"],
        "NAME OF THE ACQUIRER/DISPOSER": ["Mr X", "Mr X", "ESOP Trust", "Mrs Y"],
        "CATEGORY OF PERSON": ["Promoter Group", "Promoters", "Employees", "Director"],
        "NO. OF SECURITIES (ACQUIRED/DISPLOSED)": ["1000", "500", "10", "200"],
        "VALUE OF SECURITY (ACQUIRED/DISPLOSED)": ["1,00,000", "40000", "999", "20000"],
        "ACQUISITION/DISPOSAL TRANSACTION TYPE": ["Buy", "Sell", "Buy", "Buy"],
        "MODE OF ACQUISITION": ["Market Purchase", "Market Sale", "ESOP", "Off Market"],
        "DATE OF INITMATION TO COMPANY": ["01-11-2026", "01-11-2026", "01-11-2026", "01-11-2026"],
        "BROADCASTE DATE AND TIME": ["02-Nov-2026 18:30", "", "02-Nov-2026 10:00", "02-Nov-2026 10:00"],
    })
    df = normalise_insider(raw)
    assert list(df["symbol"]) == ["AAA", "AAA"]                 # ESOP and off-market dropped
    assert df.loc[0, "public_date"] == pd.Timestamp("2026-11-03")  # broadcast after 17:00 -> next day
    assert df.loc[1, "public_date"] == pd.Timestamp("2026-11-03")  # no broadcast -> intimation + 2 days
    assert list(df["side"]) == [1, -1] and df.loc[0, "value"] == 100000


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    sel, q = benjamini_hochberg(p, 0.05)
    assert sel.tolist() == [True, True] + [False] * 8
    assert q[0] == pytest.approx(0.01) and q[1] == pytest.approx(0.04)


def test_planted_skill_found_and_noise_rejected(small_data):
    alt = generate_synthetic_disclosures(small_data, seed=3)
    _, tab = smart_money(small_data, alt["bulk"], PARAMS)
    chosen = set(tab.index[tab["selected"]])
    smart = {f"SMART FUND {i}" for i in range(5)}
    assert smart <= chosen                                       # every genuinely skilled client is found
    lucky = chosen - smart
    assert len(lucky) / len(chosen) <= 1 / 3                     # FDR controls the share of lucky picks...
    if lucky:                                                    # ...and their short records weigh less
        assert tab.loc[list(lucky), "weight"].max() < tab.loc[list(smart), "weight"].min()
    # pure noise: shuffle which client made which deal -> nobody should survive the FDR test
    noise = alt["bulk"].copy()
    noise["client"] = [f"C{i % 40}" for i in range(len(noise))]
    noise["side"] = np.random.default_rng(0).choice([1.0, -1.0], len(noise))
    noise["symbol"] = np.random.default_rng(1).choice(small_data.symbols, len(noise))
    _, tab2 = smart_money(small_data, noise, PARAMS)
    assert tab2["selected"].sum() <= 1


def test_skill_uses_only_outcomes_known_at_the_time(small_data):
    alt = generate_synthetic_disclosures(small_data, seed=3)
    from indquant.features.smart_money import _deal_outcomes

    out = _deal_outcomes(small_data, alt["bulk"], 63)
    early = client_skill(out, as_of=100, min_deals=1, fdr_q=1.0, shrink_n=0)
    assert early["n"].sum() == ((out["known_at"] <= 100) & out["signed_excess"].notna()).sum()


def test_factors_at_t_ignore_future_prices_and_disclosures(small_data):
    alt = generate_synthetic_disclosures(small_data, seed=3)
    alt["params"] = PARAMS
    t = small_data.dates[600]
    full = MarketData(open=small_data.open, high=small_data.high, low=small_data.low, close=small_data.close,
                      volume=small_data.volume, benchmark=small_data.benchmark, index=small_data.index,
                      sectors=small_data.sectors, alt_data=alt)
    after = full.dates > t
    rng = np.random.default_rng(9)
    scale = pd.DataFrame(np.exp(rng.normal(0, .3, full.close.shape)), index=full.dates, columns=full.symbols)
    keep = np.broadcast_to(~np.asarray(after)[:, None], full.close.shape)
    bad_alt = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in alt.items()}
    late = bad_alt["bulk"]["public_date"] > t
    bad_alt["bulk"].loc[late, "client"] = "SMART FUND 0"          # rewrite the future
    bad_alt["bulk"].loc[late, "side"] = -1.0
    corrupted = MarketData(open=full.open.where(keep, full.open * scale), high=full.high.where(keep, full.high * scale),
                           low=full.low.where(keep, full.low * scale), close=full.close.where(keep, full.close * scale),
                           volume=full.volume, benchmark=full.benchmark.where(~after, 1.0), index=full.index,
                           sectors=full.sectors, alt_data=bad_alt)
    ref = smart_money_factors(full.truncate(t))
    got = smart_money_factors(corrupted)
    for k in ("smart_money", "insider_buying"):
        pd.testing.assert_series_equal(got[k].loc[t], ref[k].loc[t], check_exact=False, rtol=1e-9)


def test_import_dedupes(tmp_path):
    cfg = load_config(overrides={"smart_money.bulk_deals_file": str(tmp_path / "b.csv"),
                                 "smart_money.insider_file": str(tmp_path / "i.csv")})
    f = tmp_path / "export.csv"
    pd.DataFrame({"Date": ["03-11-2026"], "Symbol": ["TCS"], "Client Name": ["X"], "Buy/Sell": ["BUY"],
                  "Quantity Traded": ["10"], "Trade Price / Wght. Avg. Price": ["100"]}).to_csv(f, index=False)
    assert import_files(cfg, bulk=str(f)) == {"bulk": 1}
    assert import_files(cfg, bulk=str(f)) == {"bulk": 0}
