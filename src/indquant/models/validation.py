"""Leakage-free validation: walk-forward schedule and purged k-fold with embargo.

Financial labels overlap in time (a 21-day forward return computed on Monday shares 20
days with Tuesday's). Ordinary k-fold CV therefore leaks test information into training.
Purging removes training samples whose label window overlaps the test window; the
embargo additionally drops samples right after the test window (serial correlation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    fit_idx: int        # model is fitted at the close of this bar
    start_idx: int      # first bar predicted with this model (inclusive)
    end_idx: int        # last bar predicted (exclusive)


def walk_forward_windows(dates: pd.DatetimeIndex, min_train_days: int, retrain_every_months: int) -> list[Window]:
    """Expanding-window schedule: first fit after ``min_train_days`` bars, refit every N months."""
    n = len(dates)
    if min_train_days >= n:
        return []
    fit_points = [min_train_days]
    last = dates[min_train_days]
    for i in range(min_train_days + 1, n):
        d = dates[i]
        months = (d.year - last.year) * 12 + (d.month - last.month)
        if months >= retrain_every_months:
            fit_points.append(i)
            last = d
    windows = []
    for j, f in enumerate(fit_points):
        end = fit_points[j + 1] if j + 1 < len(fit_points) else n
        windows.append(Window(fit_idx=f, start_idx=f, end_idx=end))
    return windows


def purged_train_selector(sample_pos: np.ndarray, label_end_pos: np.ndarray, fit_idx: int, embargo: int) -> np.ndarray:
    """Training samples whose label was fully realised (plus embargo) by ``fit_idx``."""
    return (label_end_pos + embargo) <= fit_idx


class PurgedKFold:
    """K-fold over time-ordered samples with purging and embargo.

    ``t0`` / ``t1`` are integer bar positions where each sample's information starts
    (feature date) and where its label ends.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, t0: np.ndarray, t1: np.ndarray):
        t0 = np.asarray(t0)
        t1 = np.asarray(t1)
        order = np.argsort(t0, kind="stable")
        folds = np.array_split(order, self.n_splits)
        for test_idx in folds:
            test_start = t0[test_idx].min()
            test_end = t1[test_idx].max()
            overlap = (t1 >= test_start) & (t0 <= test_end)
            embargoed = (t0 > test_end) & (t0 <= test_end + self.embargo)
            train_mask = ~overlap & ~embargoed
            train_mask[test_idx] = False
            yield np.nonzero(train_mask)[0], np.sort(test_idx)
