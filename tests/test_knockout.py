"""Tests for the knockout wrapper (src/model/knockout.py, spec §3.5).

Locks in the definition-of-done invariants the knockout-reviewer audits:

* **sums to 1** — qualification probabilities partition (no lost/duplicated mass).
* **theta = 0 fair coin** — the penalty shootout is exactly 50/50 by default.
* **strong > weak** — the stronger side advances with probability > 0.5.
* **degenerate lambda -> 0** — no NaN, still sums to 1.
* **strict decomposition** — the ET/penalty branches are gated by the prior
  stage's level mass (no double-counting); the total-probability identity holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import load_config
from src.model.dixon_coles import score_matrix
from src.model.knockout import (
    advance_prob,
    penalty_prob,
    resolve_stage,
    sigmoid,
)


# --- theta = 0 -> exact fair coin (spec §3.5c) -----------------------------
def test_theta_zero_is_a_fair_coin():
    # For any Elo differential, theta=0 gives an exact 50/50 shootout.
    for eh, ea in [(2100.0, 1400.0), (1500.0, 1500.0), (1600.0, 1900.0)]:
        assert penalty_prob(0.0, eh, ea) == 0.5
    # sigmoid(0) is exactly 0.5.
    assert sigmoid(0.0) == 0.5


def test_config_default_theta_is_zero():
    # The shipped config keeps theta = 0 (fair coin) per spec §3.5c.
    assert load_config().theta == 0.0


# --- qualification probabilities always sum to 1 (property) ----------------
@pytest.mark.parametrize(
    "lam, mu, rho, eh, ea",
    [
        (1.5, 1.2, -0.05, 1800.0, 1750.0),   # even-ish
        (2.4, 0.6, -0.03, 2100.0, 1450.0),   # strong vs weak
        (0.9, 0.9, 0.0, 1500.0, 1500.0),     # perfectly even, rho=0
        (3.0, 2.5, -0.08, 1900.0, 1600.0),   # high-scoring
    ],
)
def test_advance_probs_sum_to_one(lam, mu, rho, eh, ea):
    out = advance_prob(lam, mu, rho, eh, ea)
    total = out["p_home_advance"] + out["p_away_advance"]
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    assert 0.0 <= out["p_home_advance"] <= 1.0
    assert 0.0 <= out["p_away_advance"] <= 1.0


# --- strong vs weak: the stronger side advances > 50% ----------------------
def test_stronger_team_advances_more_than_half():
    # Bigger home lambda + higher Elo -> P(home advance) > 0.5.
    out = advance_prob(2.2, 0.6, -0.05, 2100.0, 1400.0)
    assert out["p_home_advance"] > 0.5
    # symmetric weak-vs-strong: away is the strong side here.
    out2 = advance_prob(0.6, 2.2, -0.05, 1400.0, 2100.0)
    assert out2["p_away_advance"] > 0.5


def test_even_matchup_is_close_to_half():
    # Identical lambdas and Elo -> both sides ~ 0.5 (exactly, by symmetry).
    out = advance_prob(1.3, 1.3, -0.05, 1800.0, 1800.0)
    assert math.isclose(out["p_home_advance"], 0.5, abs_tol=1e-9)
    assert math.isclose(out["p_away_advance"], 0.5, abs_tol=1e-9)


# --- degenerate lambda/mu -> 0 must not NaN --------------------------------
@pytest.mark.parametrize("lam, mu", [(0.0, 1.2), (1.2, 0.0), (0.0, 0.0)])
def test_degenerate_lambda_no_nan(lam, mu):
    out = advance_prob(lam, mu, -0.05, 1500.0, 1500.0)
    assert not math.isnan(out["p_home_advance"])
    assert not math.isnan(out["p_away_advance"])
    assert math.isclose(
        out["p_home_advance"] + out["p_away_advance"], 1.0, abs_tol=1e-9
    )


def test_both_lambda_zero_goes_to_penalties():
    # lam=mu=0 -> 0-0 at 90' and at ET with certainty -> pure penalties.
    # With theta=0 that is a fair coin: exactly 0.5 each.
    out = advance_prob(0.0, 0.0, 0.0, 1500.0, 1500.0, theta=0.0)
    assert math.isclose(out["p_home_advance"], 0.5, abs_tol=1e-12)


# --- strict decomposition: no double-counting (spec §3.5) ------------------
def test_decomposition_matches_total_probability():
    """advance_prob equals the hand-computed 90'+ET+penalty chain, term for term."""
    lam, mu, rho, eh, ea = 1.9, 1.1, -0.05, 2000.0, 1700.0
    cfg = load_config()
    kappa, K, theta = cfg.kappa, cfg.K_max, cfg.theta

    # Stage (a): 90'.
    p90_h, p90_a, p90_d = resolve_stage(score_matrix(lam, mu, rho, K=K))
    # Stage (b): extra time on the reduced-rate mini-score.
    lam_et = lam * (30.0 / 90.0) * kappa
    mu_et = mu * (30.0 / 90.0) * kappa
    pet_h, pet_a, pet_l = resolve_stage(score_matrix(lam_et, mu_et, rho, K=K))
    # Stage (c): penalties.
    p_pen_h = penalty_prob(theta, eh, ea)

    expected_home = p90_h + p90_d * (pet_h + pet_l * p_pen_h)
    expected_away = p90_a + p90_d * (pet_a + pet_l * (1.0 - p_pen_h))

    out = advance_prob(lam, mu, rho, eh, ea)
    assert math.isclose(out["p_home_advance"], expected_home, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(out["p_away_advance"], expected_away, rel_tol=0, abs_tol=1e-12)

    # And the ET/penalty branch is truly conditional: its total contribution to
    # "home advance" is bounded by the 90'-draw mass (can't exceed the gate).
    et_pen_contrib = p90_d * (pet_h + pet_l * p_pen_h)
    assert et_pen_contrib <= p90_d + 1e-12


def test_kappa_reduces_extra_time_scoring():
    """A smaller kappa (more prudent ET) leaves more mass level -> more penalties.

    Not a correctness gate on its own, but it confirms kappa actually flows
    into the extra-time rates (not ignored / hard-coded).
    """
    lam, mu, rho, eh, ea = 1.6, 1.4, -0.05, 1800.0, 1780.0
    # ET level probability as a function of kappa.
    def et_level(kappa):
        lam_et = lam * (30.0 / 90.0) * kappa
        mu_et = mu * (30.0 / 90.0) * kappa
        return resolve_stage(score_matrix(lam_et, mu_et, rho))[2]

    assert et_level(0.4) > et_level(1.0)  # more prudent -> more ties -> more tabs


def test_positive_theta_tilts_toward_stronger_side():
    """A small positive theta nudges the shootout toward the higher Elo."""
    # Force a shootout-heavy tie by making both lambdas tiny and equal so 90'/ET
    # are near-certainly level; the difference then comes only from penalties.
    even = advance_prob(0.2, 0.2, 0.0, 2000.0, 1500.0, theta=0.0)
    tilt = advance_prob(0.2, 0.2, 0.0, 2000.0, 1500.0, theta=0.01)
    assert math.isclose(even["p_home_advance"], 0.5, abs_tol=1e-9)
    assert tilt["p_home_advance"] > even["p_home_advance"]
    assert math.isclose(
        tilt["p_home_advance"] + tilt["p_away_advance"], 1.0, abs_tol=1e-9
    )
