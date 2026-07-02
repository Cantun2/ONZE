"""Plumbing sanity checks for config.yaml + src.config.

Owned by devops-packager: these assert the config loads and centralizes the
constants the model modules depend on. Modelling-logic tests belong elsewhere.
"""

import math

from src.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg is not None


def test_core_constants_present():
    cfg = load_config()
    assert cfg.H == 100
    assert cfg.base_rating == 1500
    assert cfg.K_max == 10
    assert cfg.kappa == 0.8
    assert cfg.theta == 0.0


def test_k_by_tournament_keys():
    cfg = load_config()
    for key in ("friendly", "qualifier", "world_cup_finals", "default"):
        assert key in cfg.k_by_tournament


def test_xi_matches_documented_half_life():
    cfg = load_config()
    expected = math.log(2) / cfg.xi_half_life_days
    assert math.isclose(cfg.xi, expected, rel_tol=1e-9)


def test_paths_are_absolute():
    cfg = load_config()
    assert cfg.paths.raw_dir.is_absolute()
    assert cfg.paths.processed_dir.is_absolute()
    assert cfg.paths.db_path.is_absolute()
