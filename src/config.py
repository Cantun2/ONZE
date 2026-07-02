"""Central configuration loader for ONZE.

Every module imports its constants from here, so there are no magic numbers
scattered in code (CLAUDE.md prime directive #4). The values live in the
repo-root ``config.yaml``; this module reads them once into a frozen dataclass.

Usage::

    from src.config import load_config
    cfg = load_config()
    cfg.H            # 100
    cfg.xi           # per-day time-decay rate
    cfg.paths.db_path  # resolved absolute Path

This file is plumbing only: it contains no modelling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Repo root = parent of the ``src`` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved to absolute paths against the repo root."""

    raw_dir: Path
    processed_dir: Path
    db_path: Path


@dataclass(frozen=True)
class Config:
    """Typed view over ``config.yaml``.

    Attributes mirror the YAML keys documented in ``config.yaml`` and in
    ONZE_spec.md §3. ``raw`` keeps the untouched dict for any key not promoted
    to a typed field yet.
    """

    k_by_tournament: dict[str, float]
    H: float
    base_rating: float
    xi: float
    xi_half_life_days: float
    K_max: int
    kappa: float
    theta: float
    paths: Paths
    raw: dict


def _resolve(root: Path, value: str) -> Path:
    """Resolve a possibly-relative path from config against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else (root / p)


@lru_cache(maxsize=None)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load ``config.yaml`` into a :class:`Config`.

    Cached per path so repeated imports are cheap. Pass ``config_path`` to
    override the default (useful in tests).
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    root = path.resolve().parent
    raw_paths = data.get("paths", {})
    paths = Paths(
        raw_dir=_resolve(root, raw_paths["raw_dir"]),
        processed_dir=_resolve(root, raw_paths["processed_dir"]),
        db_path=_resolve(root, raw_paths["db_path"]),
    )

    return Config(
        k_by_tournament=dict(data["k_by_tournament"]),
        H=float(data["H"]),
        base_rating=float(data["base_rating"]),
        xi=float(data["xi"]),
        xi_half_life_days=float(data["xi_half_life_days"]),
        K_max=int(data["K_max"]),
        kappa=float(data["kappa"]),
        theta=float(data["theta"]),
        paths=paths,
        raw=data,
    )


if __name__ == "__main__":
    # Small smoke helper: `python -m src.config` prints the resolved config.
    import json

    cfg = load_config()
    print(
        json.dumps(
            {
                "k_by_tournament": cfg.k_by_tournament,
                "H": cfg.H,
                "base_rating": cfg.base_rating,
                "xi": cfg.xi,
                "xi_half_life_days": cfg.xi_half_life_days,
                "K_max": cfg.K_max,
                "kappa": cfg.kappa,
                "theta": cfg.theta,
                "paths": {
                    "raw_dir": str(cfg.paths.raw_dir),
                    "processed_dir": str(cfg.paths.processed_dir),
                    "db_path": str(cfg.paths.db_path),
                },
            },
            indent=2,
        )
    )
