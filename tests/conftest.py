import pytest

from indquant.config import load_config
from indquant.data import generate_synthetic_market


def small_overrides(tmp_path=None, **extra):
    o = {
        "data.source": "synthetic",
        "data.synthetic.n_stocks": 60,
        "data.synthetic.n_days": 900,
        "universe_filter.min_adv_inr": 1e6,
        "universe_filter.min_history_days": 150,
        "ml.min_train_years": 1.0,
        "ml.retrain_every_months": 6,
        "ml.n_estimators": 40,
        "ml.n_jobs": 1,
        "ml.min_child_samples": 50,
        "general.backtest_start": None,
        "validation.bootstrap_reps": 0,
        "swing.universe_top_liquid": 40,
    }
    if tmp_path is not None:
        o.update({
            "general.output_dir": str(tmp_path / "output"),
            "paper.db_path": str(tmp_path / "state" / "paper.db"),
            "logging.dir": str(tmp_path / "logs"),
            "logging.trial_log": str(tmp_path / "state" / "trials.csv"),
            "logging.expectations_file": str(tmp_path / "state" / "expectations.json"),
            "sentiment.history_file": str(tmp_path / "sentiment.csv"),
        })
    o.update(extra)
    return o


@pytest.fixture
def small_cfg(tmp_path):
    return load_config(overrides=small_overrides(tmp_path))


@pytest.fixture(scope="session")
def small_data():
    return generate_synthetic_market(n_stocks=60, n_days=900, seed=7, alpha_strength=1.0)


@pytest.fixture(autouse=True)
def _reset_logging():
    """The CLI installs root log handlers bound to pytest's captured streams - remove them."""
    import logging

    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
