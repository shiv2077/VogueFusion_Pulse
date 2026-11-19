"""Load configs/voguefusion.yaml and resolve paths."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from costing.solver import SolverParams

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "voguefusion.yaml"


def load(path: str | None = None) -> dict[str, Any]:
    """Parsed config. VOGUEFUSION_CONFIG overrides the default path."""
    return _load(str(Path(path or os.environ.get("VOGUEFUSION_CONFIG") or DEFAULT_CONFIG)))


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def data_dir(cfg: dict[str, Any]) -> Path:
    """Runtime data root: $VOGUEFUSION_DATA, else the configured default. Created if missing."""
    d = Path(os.environ.get(cfg["data_dir_env"]) or cfg["data_dir_default"]).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_path(relative: str) -> Path:
    """A path from the config, relative to the repository root unless absolute or ~."""
    p = Path(relative).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def solver_params(cfg: dict[str, Any]) -> SolverParams:
    s = dict(cfg["costing"]["solver"])
    s.pop("version")
    return SolverParams(**s)
