"""Beginner setup path: example settings file and automatic config discovery."""
from pathlib import Path

from indquant.cli import resolve_config_path
from indquant.config import REPO_ROOT, load_config


def test_example_settings_load_cleanly():
    cfg = load_config(REPO_ROOT / "config" / "example.yaml")
    assert cfg.general.initial_capital == 500000 and cfg.data.source == "nse"
    assert cfg.core.top_n == 20 and cfg.execution.broker == "manual"


def test_config_discovery(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INDQUANT_CONFIG", raising=False)
    assert resolve_config_path(None) is None
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "my.yaml").write_text("general:\n  initial_capital: 123\n")
    assert resolve_config_path(None) == str(Path("config") / "my.yaml")
    monkeypatch.setenv("INDQUANT_CONFIG", "/x/other.yaml")
    assert resolve_config_path(None) == "/x/other.yaml"
    assert resolve_config_path("explicit.yaml") == "explicit.yaml"


def test_expected_session_and_missing_today_blocks_signals(small_data):
    from datetime import datetime, timedelta

    from indquant.live.healthcheck import check_data, expected_last_session, has_errors

    assert str(expected_last_session(datetime(2026, 11, 2, 20, 0))) == "2026-11-02"       # Monday evening
    assert str(expected_last_session(datetime(2026, 11, 2, 11, 0))) == "2026-10-30"       # Monday morning -> Friday
    assert str(expected_last_session(datetime(2026, 11, 1, 20, 0))) == "2026-10-30"       # Sunday -> Friday
    assert str(expected_last_session(datetime(2026, 11, 2, 20, 0), ["2026-11-02"])) == "2026-10-30"

    cfg = load_config()
    last = small_data.dates[-1]
    small_data.source = "nse"
    try:
        evening_after = datetime(last.year, last.month, last.day, 20, 0) + timedelta(days=1)
        while evening_after.weekday() >= 5:
            evening_after += timedelta(days=1)
        issues = check_data(small_data, cfg, now=evening_after)
        assert has_errors(issues) and any("not published yet" in i.message for i in issues)
        same_evening = datetime(last.year, last.month, last.day, 20, 0)
        assert not any("not published" in i.message for i in check_data(small_data, cfg, now=same_evening))
    finally:
        small_data.source = "synthetic"


def test_nse_daily_load_fetches_only_new_days(tmp_path):
    from datetime import date

    import indquant.data.nse_bhavcopy as nb

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "20240102.csv.zip").write_bytes(b"x")
    calls = []

    def fake_download(start, end, raw_dir, fetch=None, pause=0):
        calls.append((start, end))
        return {}

    cfg = load_config(overrides={"nse.raw_dir": str(raw), "general.start": "2024-01-01", "general.end": "2024-01-10"})
    orig = nb.download_range
    nb.download_range = fake_download
    try:
        try:
            nb.load_nse_panel(cfg)
        except Exception:
            pass  # the dummy file is not a real bhavcopy; only the download range matters here
    finally:
        nb.download_range = orig
    assert calls == [(date(2024, 1, 3), date(2024, 1, 10))]
