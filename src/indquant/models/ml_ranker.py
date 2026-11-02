"""Walk-forward gradient boosting models on cross-sectional features.

Every prediction at date t comes from a model fitted only on samples whose labels were
fully observable (plus an embargo) before the model's fit date. The backtest and the live
signal generator use exactly the same schedule, so live signals are reproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .validation import purged_train_selector, walk_forward_windows

log = logging.getLogger(__name__)


def stack_features(features: dict[str, pd.DataFrame], rows: pd.DatetimeIndex, mask: pd.DataFrame):
    """Long matrix for (date, symbol) pairs where ``mask`` is True. Returns (X, row_pos, col_pos)."""
    m = mask.loc[rows].to_numpy(dtype=bool)
    ri, ci = np.nonzero(m)
    cols = {}
    for name, f in features.items():
        arr = f.reindex(index=rows, columns=mask.columns).to_numpy(dtype=np.float32)
        cols[name] = arr[ri, ci]
    X = pd.DataFrame(cols)
    return X, ri, ci


def _make_model(kind: str, ml_cfg, seed: int):
    try:
        import lightgbm as lgb

        params = dict(
            n_estimators=int(ml_cfg.n_estimators), learning_rate=float(ml_cfg.learning_rate),
            num_leaves=int(ml_cfg.num_leaves), min_child_samples=int(ml_cfg.min_child_samples),
            subsample=float(ml_cfg.subsample), subsample_freq=1, colsample_bytree=float(ml_cfg.colsample_bytree),
            reg_lambda=float(ml_cfg.reg_lambda), random_state=seed, n_jobs=int(ml_cfg.n_jobs), verbose=-1,
        )
        return lgb.LGBMClassifier(**params) if kind == "classifier" else lgb.LGBMRegressor(**params)
    except ImportError:  # pragma: no cover - fallback when lightgbm is unavailable
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        params = dict(max_iter=int(ml_cfg.n_estimators), learning_rate=float(ml_cfg.learning_rate),
                      max_leaf_nodes=int(ml_cfg.num_leaves), min_samples_leaf=int(ml_cfg.min_child_samples),
                      random_state=seed)
        return HistGradientBoostingClassifier(**params) if kind == "classifier" else HistGradientBoostingRegressor(**params)


@dataclass
class WalkForwardModel:
    kind: str                      # "regressor" | "classifier"
    ml_cfg: dict
    horizon: int                   # bars until the label is fully known (beyond t+1)
    sample_every: int = 5
    seed: int = 42
    min_samples: int = 2000
    fit_dates: list = field(default_factory=list)
    importance: pd.DataFrame | None = None

    def fit_predict(self, features: dict[str, pd.DataFrame], label: pd.DataFrame, mask: pd.DataFrame,
                    predict_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        dates = mask.index
        out = pd.DataFrame(np.nan, index=dates, columns=mask.columns)
        min_train = int(float(self.ml_cfg.min_train_years) * 252)
        windows = walk_forward_windows(dates, min_train, int(self.ml_cfg.retrain_every_months))
        if not windows:
            log.info("Not enough history for ML (%d bars, need %d) - factor/rule scores only", len(dates), min_train)
            return out

        train_rows = dates[:: self.sample_every]
        train_mask = mask & label.notna()
        X_all, ri, ci = stack_features(features, train_rows, train_mask)
        y_all = label.reindex(index=train_rows, columns=mask.columns).to_numpy()[ri, ci]
        sample_pos = dates.get_indexer(train_rows)[ri]
        label_end = sample_pos + 1 + self.horizon
        embargo = int(self.ml_cfg.embargo_days)

        pred_pos = np.arange(len(dates)) if predict_dates is None else np.sort(dates.get_indexer(predict_dates))
        pred_pos = pred_pos[pred_pos >= 0]
        importances = []
        for w in windows:
            rows_pos = pred_pos[(pred_pos >= w.start_idx) & (pred_pos < w.end_idx)]
            if len(rows_pos) == 0:
                continue
            sel = purged_train_selector(sample_pos, label_end, w.fit_idx, embargo)
            if sel.sum() < self.min_samples:
                continue
            y = y_all[sel]
            model = _make_model(self.kind, self.ml_cfg, self.seed)
            if self.kind == "classifier":
                if len(np.unique(y)) < 2:
                    continue
                model.fit(X_all[sel], y.astype(int))
            else:
                model.fit(X_all[sel], y)
            self.fit_dates.append(dates[w.fit_idx])
            if hasattr(model, "feature_importances_"):
                importances.append(pd.Series(model.feature_importances_, index=X_all.columns, name=dates[w.fit_idx]))

            rows = dates[rows_pos]
            Xp, pri, pci = stack_features(features, rows, mask)
            if len(Xp) == 0:
                continue
            pred = model.predict_proba(Xp)[:, 1] if self.kind == "classifier" else model.predict(Xp)
            block = np.full((len(rows), mask.shape[1]), np.nan)
            block[pri, pci] = pred
            out.iloc[rows_pos] = block
        if importances:
            self.importance = pd.DataFrame(importances)
        return out
