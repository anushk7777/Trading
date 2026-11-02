"""Configuration loading.

The canonical defaults live in ``config/default.yaml``. A user config file is merged on
top of the defaults, then ``--set section.key=value`` overrides are applied. Unknown keys
raise an error so that typos never silently fall back to a default.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """A dict with attribute access for nested sections (``cfg.core.top_n``)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute errors surface as normal
            raise AttributeError(item) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def to_dict(self) -> dict:
        return _to_plain(self)

    def copy_with(self, overrides: dict[str, Any] | None = None) -> "Config":
        new = copy.deepcopy(self)
        for key, value in (overrides or {}).items():
            set_dotted(new, key, value, strict=True)
        validate(new)
        return new


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _merge(base: dict, override: dict, path: str = "", strict: bool = True) -> dict:
    for key, value in override.items():
        full = f"{path}.{key}" if path else key
        if strict and key not in base and not _is_free_form(path):
            raise KeyError(f"Unknown config key: {full}")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, full, strict)
        else:
            base[key] = value
    return base


def _is_free_form(path: str) -> bool:
    # factor weights may contain user-defined factors
    return path.endswith("factor_weights")


def parse_value(raw: str) -> Any:
    """Parse a CLI override value using YAML rules (numbers, bools, null, lists)."""
    return yaml.safe_load(raw)


def set_dotted(cfg: dict, dotted: str, value: Any, strict: bool = True) -> None:
    parts = dotted.split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node:
            if strict:
                raise KeyError(f"Unknown config section: {'.'.join(parts[: i + 1])}")
            node[part] = Config()
        node = node[part]
    leaf = parts[-1]
    if strict and leaf not in node and not _is_free_form(".".join(parts[:-1])):
        raise KeyError(f"Unknown config key: {dotted}")
    node[leaf] = _wrap(value)


def get_dotted(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def validate(cfg: Config) -> None:
    alloc = cfg.allocation
    if alloc.core < 0 or alloc.swing < 0 or alloc.core + alloc.swing > 1.0 + 1e-9:
        raise ValueError("allocation.core + allocation.swing must be within [0, 1]")
    if cfg.core.top_n < 1:
        raise ValueError("core.top_n must be >= 1")
    if cfg.core.buffer_mult < 1.0:
        raise ValueError("core.buffer_mult must be >= 1")
    if not 0 < cfg.core.max_weight <= 1:
        raise ValueError("core.max_weight must be in (0, 1]")
    if cfg.core.construction not in ("optimizer", "heuristic"):
        raise ValueError("core.construction must be 'optimizer' or 'heuristic'")
    if cfg.core.weighting not in ("equal", "inverse_vol"):
        raise ValueError("core.weighting must be 'equal' or 'inverse_vol'")
    if cfg.swing.stop_atr <= 0 or cfg.swing.target_atr <= 0:
        raise ValueError("swing.stop_atr and swing.target_atr must be positive")
    if cfg.data.source not in ("yahoo", "csv", "synthetic", "nse"):
        raise ValueError("data.source must be yahoo, nse, csv or synthetic")
    if not 0 <= cfg.core.ml_weight <= 1:
        raise ValueError("core.ml_weight must be in [0, 1]")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        _merge(base, user)
    cfg = _wrap(base)
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value, strict=True)
    validate(cfg)
    return cfg


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item}")
        key, raw = item.split("=", 1)
        out[key.strip()] = parse_value(raw.strip())
    return out
