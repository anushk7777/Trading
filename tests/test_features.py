import numpy as np
import pandas as pd
import pytest

from indquant.features.factors import compute_factors, rolling_beta
from indquant.features.preprocessing import combine, cs_rank, sector_neutralize
from indquant.features.technical import atr, rsi


def test_rsi_extremes():
    up = pd.DataFrame({"A": np.arange(1, 60, dtype=float)})
    down = pd.DataFrame({"A": np.arange(60, 1, -1, dtype=float)})
    assert rsi(up).iloc[-1, 0] == pytest.approx(100.0)
    assert rsi(down).iloc[-1, 0] == pytest.approx(0.0)


def test_atr_constant_range():
    idx = pd.RangeIndex(50)
    close = pd.DataFrame({"A": 100.0}, index=idx)
    high, low = close + 2, close - 2
    assert atr(high, low, close, 14).iloc[-1, 0] == pytest.approx(4.0)


def test_cs_rank_centered_and_masked():
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=list("abcd"))
    mask = pd.DataFrame([[True, True, True, False]], columns=list("abcd"))
    r = cs_rank(df, mask)
    assert np.isnan(r.iloc[0, 3])
    assert r.iloc[0, 0] < r.iloc[0, 1] < r.iloc[0, 2]
    assert r.iloc[0, :3].sum() == pytest.approx(0.0)
    assert r.abs().max().max() < 0.5


def test_sector_neutralize_zero_mean_within_sector():
    df = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], columns=list("abcd"))
    sec = pd.Series(["x", "x", "y", "y"], index=list("abcd"))
    out = sector_neutralize(df, sec)
    assert out[["a", "b"]].sum(axis=1).iloc[0] == pytest.approx(0)
    assert out[["c", "d"]].sum(axis=1).iloc[0] == pytest.approx(0)


def test_combine_renormalises_missing_scores():
    a = pd.DataFrame([[1.0, np.nan]], columns=["x", "y"])
    b = pd.DataFrame([[0.0, 2.0]], columns=["x", "y"])
    out = combine({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    assert out.iloc[0, 0] == pytest.approx(0.5)
    assert out.iloc[0, 1] == pytest.approx(2.0)


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 400))
    ret = pd.DataFrame({"A": 1.5 * m + rng.normal(0, 0.001, 400)})
    assert rolling_beta(ret, m).iloc[-1, 0] == pytest.approx(1.5, abs=0.05)


def test_momentum_skips_last_month(small_data):
    f = compute_factors(small_data)
    c = small_data.close
    t = 400
    expected = c.iloc[t - 21] / c.iloc[t - 252] - 1
    pd.testing.assert_series_equal(f["mom_12_1"].iloc[t], expected, check_names=False)
