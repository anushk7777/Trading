"""Corporate-action detection must not mistake data gaps for splits (real NSE pitfalls)."""
import urllib.error
from datetime import date

import numpy as np
import pandas as pd

from indquant.data.nse_bhavcopy import SPECIAL_SESSIONS, bhavcopy_url, detect_actions, download_range


def _panel(n_days=60, n_sym=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_sym)), axis=0)), index=idx, columns=cols)
    prev = close.shift(1)
    vol = pd.DataFrame(1e5, index=idx, columns=cols)
    return close, prev, close * 0.999, vol


def test_missing_session_is_not_thousands_of_splits_but_a_real_split_survives():
    close, _, _, vol = _panel()
    d = close.index[30]
    close.loc[d:, "S00"] *= 0.5                                     # a genuine 1:1 bonus on that Monday
    prev = close.shift(1)
    rng = np.random.default_rng(1)
    session_move = pd.Series(np.exp(rng.normal(0, 0.05, close.shape[1])), index=close.columns)
    prev.loc[d] = close.iloc[29] * session_move                  # PREVCLOSE is the unseen session's close
    prev.loc[d, "S00"] = close.iloc[29]["S00"] * session_move["S00"] * 0.5
    ev = detect_actions(close, prev, close * 0.999, vol)
    assert list(ev["symbol"]) == ["S00"] and abs(ev["factor"].iloc[0] / session_move["S00"] - 0.5) < 1e-9


def test_stock_returning_after_weeks_away_keeps_its_real_loss():
    close, _, _, vol = _panel()
    close.iloc[20:40, 0] = np.nan                                   # away (e.g. moved to a series we don't load)
    close.iloc[40:, 0] = close.iloc[19, 0] * 0.62 * close.iloc[40:, 0] / close.iloc[40, 0]
    prev = close.ffill().shift(1)
    prev.iloc[40, 0] = close.iloc[19, 0] * 0.62                     # fell 38% while away: real loss, not a split
    open_ = close * 0.999
    assert detect_actions(close, prev, open_, vol).empty
    prev.iloc[40, 0] = close.iloc[19, 0] * 0.5                      # an exact 1:2 ratio is still accepted
    assert detect_actions(close, prev, open_, vol)["symbol"].tolist() == ["S00"]


def test_known_weekend_sessions_are_downloaded(tmp_path):
    asked = []

    def fetch(url):
        asked.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    download_range(date(2020, 1, 31), date(2020, 2, 3), tmp_path, fetch=fetch, pause=0)
    assert date(2020, 2, 1) in SPECIAL_SESSIONS
    assert bhavcopy_url(date(2020, 2, 1)) in asked and bhavcopy_url(date(2020, 2, 2)) not in asked
