"""Free index/ETF feeds sometimes carry impossible one-day moves; they must not reach the statistics."""
import numpy as np
import pandas as pd

from indquant.data.base import clean_index_series


def _series(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)


def test_one_day_spike_is_removed():
    s = _series()
    bad = s.copy()
    bad.iloc[150] *= 10                                       # a x10 bad tick that reverts the next day
    fixed, fixes = clean_index_series(bad)
    r = fixed.pct_change().dropna()
    assert r.abs().max() < 0.1 and len(fixes) == 1 and "bad price" in fixes[0]
    assert fixed.iloc[150] == bad.iloc[149]
    assert np.allclose(fixed.drop(fixed.index[150]), s.drop(s.index[150]))
    dd = (fixed / fixed.cummax() - 1).min()
    assert dd > -0.5                                          # no fake -90% drawdown


def test_unadjusted_unit_split_is_back_adjusted():
    s = _series()
    bad = s.copy()
    bad.iloc[:200] *= 10                                      # 1:10 unit split missing from history
    fixed, fixes = clean_index_series(bad)
    assert len(fixes) == 1 and "split" in fixes[0]
    assert np.allclose(fixed / fixed.iloc[-1], s / s.iloc[-1])


def test_clean_series_is_untouched_and_nans_kept():
    s = _series()
    s.iloc[[10, 11, 50]] = np.nan
    fixed, fixes = clean_index_series(s)
    assert not fixes and fixed.equals(s.astype(float))
