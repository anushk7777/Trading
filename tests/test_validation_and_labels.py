import numpy as np
import pandas as pd

from indquant.data.base import MarketData
from indquant.models.labeling import triple_barrier
from indquant.models.validation import PurgedKFold, purged_train_selector, walk_forward_windows


def test_purged_kfold_has_no_overlap_and_respects_embargo():
    n, h, emb = 300, 10, 5
    t0 = np.arange(n)
    t1 = t0 + h
    for train, test in PurgedKFold(n_splits=5, embargo=emb).split(t0, t1):
        lo, hi = t0[test].min(), t1[test].max()
        assert not np.any((t1[train] >= lo) & (t0[train] <= hi)), "train label windows overlap test"
        assert not np.any((t0[train] > hi) & (t0[train] <= hi + emb)), "embargo violated"
        assert len(np.intersect1d(train, test)) == 0


def test_walk_forward_training_never_sees_unrealised_labels():
    dates = pd.bdate_range("2018-01-01", periods=1000)
    windows = walk_forward_windows(dates, 252, 6)
    assert windows[0].fit_idx == 252
    assert all(w.start_idx == w.fit_idx for w in windows)
    pos = np.arange(1000)
    for w in windows:
        sel = purged_train_selector(pos, pos + 1 + 21, w.fit_idx, 10)
        assert (pos[sel] + 1 + 21 + 10 <= w.fit_idx).all()


def _one_stock(o, h, l, c):
    idx = pd.bdate_range("2020-01-01", periods=len(c))
    f = lambda v: pd.DataFrame({"S": v}, index=idx)  # noqa: E731
    return MarketData(open=f(o), high=f(h), low=f(l), close=f(c), volume=f([1e6] * len(c)),
                      benchmark=pd.Series(c, index=idx), index=pd.Series(c, index=idx))


def test_triple_barrier_target_stop_and_ambiguous_bar():
    # bar0 signal, bar1 entry @100. ATR=1, stop=98, target=103
    o = [100, 100, 101, 102, 100, 100]
    h = [100, 101, 102, 103.5, 101, 101]
    l = [100, 99.5, 100, 101, 99, 99]
    c = [100, 100.5, 101.5, 103, 100, 100]
    d = _one_stock(o, h, l, c)
    atr = pd.DataFrame({"S": [1.0] * 6}, index=d.dates)
    tb = triple_barrier(d, atr, 2.0, 3.0, 4)
    assert tb.label.iloc[0, 0] == 1.0 and tb.exit_offset.iloc[0, 0] == 3
    assert tb.ret.iloc[0, 0] == np.float64(103 / 100 - 1)

    # same bar touches both -> stop assumed first
    h2 = [100, 104, 100, 100, 100, 100]
    l2 = [100, 97, 100, 100, 100, 100]
    tb2 = triple_barrier(_one_stock(o, h2, l2, c), atr, 2.0, 3.0, 4)
    assert tb2.label.iloc[0, 0] == 0.0 and tb2.exit_offset.iloc[0, 0] == 1


def test_triple_barrier_unknown_future_is_nan():
    o = h = l = c = [100.0] * 5
    d = _one_stock(o, h, l, c)
    atr = pd.DataFrame({"S": [1.0] * 5}, index=d.dates)
    tb = triple_barrier(d, atr, 2.0, 3.0, 10)
    assert tb.label.isna().all().all()
