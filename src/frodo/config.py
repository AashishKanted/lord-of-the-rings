"""Loads ``config/frodo.toml`` once and exposes it with attribute access.

    from frodo.config import get_config
    cfg = get_config()
    cfg.navigation.max_linear_mps
"""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "frodo.toml"


class Section(dict):
    """A dict whose keys can also be read as attributes; unknown keys fail loudly."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key '{key}' (have: {sorted(self)})") from None


def _wrap(value):
    if isinstance(value, dict):
        return Section({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


@lru_cache(maxsize=None)
def _load(path: Path) -> Section:
    with path.open("rb") as f:
        return _wrap(tomllib.load(f))


def get_config(path: str | Path | None = None) -> Section:
    """Explicit path > $FRODO_CONFIG > config/frodo.toml."""
    return _load(Path(path or os.environ.get("FRODO_CONFIG") or DEFAULT_CONFIG).resolve())


def with_overrides(config: Section, overrides: dict) -> Section:
    """A copy of ``config`` with nested values replaced, e.g.
    ``with_overrides(cfg, {"plant": {"release_lead_m": 0.3}})``. The original is untouched."""
    def merge(base, extra):
        out = Section(base)
        for key, value in extra.items():
            if key not in base:
                raise KeyError(f"config has no key '{key}' (have: {sorted(base)})")
            out[key] = merge(base[key], value) if isinstance(value, dict) else _wrap(value)
        return out
    return merge(config, overrides)
