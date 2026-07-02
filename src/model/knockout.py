"""Knockout phase: extra time + penalty shootout -> P(qualification) (spec §3.5).

In a knockout tie there is no draw: the winner must be decided. ONZE resolves
this by chaining **three strictly-conditional stages**, each entered only if the
previous one was level (spec §3.5). This module is a *wrapper* around the base
90' goals model (:mod:`src.model.dixon_coles`) — it never rewrites the score
matrix, it consumes it.

The three stages (spec §3.5)
----------------------------
**(a) Regulation, 90'.** Build ``P_90(x, y) = score_matrix(lam, mu, rho, K)``.
::

    P_90(home) = sum_{x>y} P_90(x, y)     # home win in regulation
    P_90(away) = sum_{x<y} P_90(x, y)     # away win in regulation
    P_90(draw) = sum_{x=y} P_90(x, y)     # level after 90' -> go to extra time

If ``x != y`` the tie is resolved here; the ET/penalty branches are entered
**only** through the ``P_90(draw)`` mass, so there is no double-counting.

**(b) Extra time (conditional on a 90' draw).** 30 extra minutes, played more
prudently and with fatigue, so the goal rates are scaled down (spec §3.5)::

    lambda_ET = lam * (30 / 90) * kappa,   mu_ET = mu * (30 / 90) * kappa

with ``kappa ~= 0.8`` (prudence factor, from config). We re-use the very same
Dixon-Coles primitive on this *extra-time mini-score*::

    P_ET(x, y) = score_matrix(lambda_ET, mu_ET, rho, K)
    P_ET(home) = sum_{x>y} P_ET,  P_ET(away) = sum_{x<y} P_ET,
    P_ET(level) = sum_{x=y} P_ET  # still level after ET -> penalties

**(c) Penalties (conditional on still level after ET).** Historically ~50/50,
only weakly tied to strength (spec §3.5)::

    P_pen(home) = sigmoid(theta * (elo_home - elo_away))

``theta`` is a small config parameter; the default ``theta = 0`` gives an exact
fair coin (0.5 / 0.5), independent of Elo. A helper,
:func:`calibrate_theta`, can optionally fit ``theta`` on the historical shootout
CSV, but the default path uses the config value.

Total probability of advancing (spec §3.5)
------------------------------------------
::

    P(home advance) = P_90(home)
                    + P_90(draw) * [ P_ET(home) + P_ET(level) * P_pen(home) ]

and symmetrically for away with ``P_pen(away) = 1 - P_pen(home)``. Because every
partition sums to 1 at each stage, ``p_home_advance + p_away_advance == 1``
exactly (up to float rounding) — no probability mass is lost or double-counted.

Contract
--------
* Consumes ``score_matrix`` / ``tau`` from :mod:`src.model.dixon_coles` and
  ``predict_lambdas`` from :mod:`src.model.lambdas`; it does **not** change the
  base goals model.
* ``kappa``, ``theta``, ``K`` are parameters (default to
  :func:`src.config.load_config`), never magic numbers buried in code.
* Numerically safe when ``lam`` or ``mu`` -> 0 (the score matrix already puts
  all mass on the 0-goal row/column without NaN; ET rates just shrink to 0).

Public API
----------
``sigmoid``          numerically-stable logistic.
``penalty_prob``     ``P_pen(home)`` from ``theta`` and the Elo differential.
``resolve_stage``    (home, away, level) triple from a score matrix.
``advance_prob``     the full 90'+ET+penalty chain -> qualification probs.
``calibrate_theta``  optional MLE of ``theta`` on historical shootouts.
"""

from __future__ import annotations

import math

import numpy as np

from src.config import load_config
from src.model.dixon_coles import score_matrix

# Extra time is 30 of the 90 regulation minutes (spec §3.5). Not a tunable knob,
# it is the fixed duration ratio of the laws of the game — kept named for clarity.
_ET_MINUTES_RATIO = 30.0 / 90.0


# ---------------------------------------------------------------------------
# Penalty shootout: fair-coin sigmoid on the Elo differential (spec §3.5c)
# ---------------------------------------------------------------------------
def sigmoid(z: float) -> float:
    """Numerically-stable logistic ``sigma(z) = 1 / (1 + e^-z)``.

    Split by the sign of ``z`` so neither ``exp`` overflows for large ``|z|``.
    """
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def penalty_prob(theta: float, elo_home: float, elo_away: float) -> float:
    """``P_pen(home) = sigmoid(theta * (elo_home - elo_away))`` (spec §3.5c).

    With ``theta = 0`` (the config default) this is exactly ``0.5`` for any Elo
    differential — a fair coin, as the spec prescribes ("ne pas surinterpréter").
    A small positive ``theta`` tilts the shootout very slightly toward the
    stronger side.
    """
    return sigmoid(theta * (elo_home - elo_away))


# ---------------------------------------------------------------------------
# Resolve a score matrix into (home-win, away-win, level) (spec §3.4/§3.5)
# ---------------------------------------------------------------------------
def resolve_stage(P: np.ndarray) -> tuple[float, float, float]:
    """Split a score matrix into ``(p_home, p_away, p_level)`` (spec §3.4).

    ``P[x, y]`` is ``P(home scores x, away scores y)``. Home wins on ``x > y``
    (strictly-lower triangle), away on ``x < y`` (strictly-upper triangle), and
    the stage is level on the diagonal ``x == y``. The three sum to ``P.sum()``,
    which is 1 for a (renormalised) ``score_matrix``.
    """
    p_home = float(np.tril(P, -1).sum())  # x > y
    p_away = float(np.triu(P, 1).sum())   # x < y
    p_level = float(np.trace(P))          # x == y
    return p_home, p_away, p_level


# ---------------------------------------------------------------------------
# The full knockout chain (spec §3.5)
# ---------------------------------------------------------------------------
def advance_prob(
    lam: float,
    mu: float,
    rho: float,
    elo_home: float,
    elo_away: float,
    theta: float | None = None,
    kappa: float | None = None,
    K: int | None = None,
) -> dict[str, float]:
    """Qualification probabilities for a knockout tie (spec §3.5).

    Chains three strictly-conditional stages — 90', extra time, penalties — and
    returns ``{"p_home_advance": ..., "p_away_advance": ...}``. The two always
    sum to 1 (float tolerance), because each stage is a proper partition and the
    ET/penalty branches are entered *only* through the 90'-draw mass (no
    double-counting).

    Parameters
    ----------
    lam, mu, rho
        The base 90' Dixon-Coles goal parameters for this fixture — exactly the
        inputs to :func:`src.model.dixon_coles.score_matrix`. Not re-derived
        here; pass what the goals model produced.
    elo_home, elo_away
        Elo ratings, used *only* by the penalty sigmoid (spec §3.5c).
    theta, kappa, K
        Config parameters (spec §3.5): ``theta`` = penalty sigmoid slope
        (default ``0`` -> fair coin), ``kappa`` = extra-time prudence factor
        (~0.8), ``K`` = score-matrix truncation. ``None`` -> read from
        :func:`src.config.load_config` so they are never magic numbers.

    Notes
    -----
    Degenerate ``lam`` or ``mu`` -> 0 is safe: ``score_matrix`` collapses the
    mass onto the 0-goal row/column without NaN, and the extra-time rates scale
    to 0 (a 0-0 ET, i.e. certain to go to penalties) rather than misbehaving.
    """
    if theta is None or kappa is None or K is None:
        cfg = load_config()
        if theta is None:
            theta = cfg.theta
        if kappa is None:
            kappa = cfg.kappa
        if K is None:
            K = cfg.K_max

    # --- (a) Regulation, 90' (spec §3.5a) ---------------------------------
    P90 = score_matrix(lam, mu, rho, K=K)
    p90_home, p90_away, p90_draw = resolve_stage(P90)

    # --- (b) Extra time, conditional on a 90' draw (spec §3.5b) -----------
    # Reduced rates: 30/90 of the match played more prudently (kappa).
    lam_et = lam * _ET_MINUTES_RATIO * kappa
    mu_et = mu * _ET_MINUTES_RATIO * kappa
    P_et = score_matrix(lam_et, mu_et, rho, K=K)
    pet_home, pet_away, pet_level = resolve_stage(P_et)

    # --- (c) Penalties, conditional on still level after ET (spec §3.5c) ---
    p_pen_home = penalty_prob(theta, elo_home, elo_away)
    p_pen_away = 1.0 - p_pen_home

    # --- Total probability of advancing (spec §3.5) -----------------------
    # Extra-time and penalty terms are gated by the prior stage's level mass,
    # so this is a strict decomposition (no double-counting).
    p_home_advance = p90_home + p90_draw * (pet_home + pet_level * p_pen_home)
    p_away_advance = p90_away + p90_draw * (pet_away + pet_level * p_pen_away)

    return {"p_home_advance": p_home_advance, "p_away_advance": p_away_advance}


# ---------------------------------------------------------------------------
# Optional: calibrate theta on the historical shootout record (spec §3.5c)
# ---------------------------------------------------------------------------
def calibrate_theta(
    elo_diffs: np.ndarray,
    home_won: np.ndarray,
    theta_bounds: tuple[float, float] = (-0.02, 0.02),
) -> float:
    """MLE of the penalty slope ``theta`` on historical shootouts (spec §3.5c).

    Given per-shootout Elo differentials ``elo_home - elo_away`` and a binary
    ``home_won`` outcome, maximise the Bernoulli log-likelihood of
    ``P_pen(home) = sigmoid(theta * diff)`` over a *small* bounded ``theta`` (the
    spec warns against over-interpreting: shootouts are close to a fair coin).

    This is **optional** — the default :func:`advance_prob` path uses the config
    ``theta`` (0). It is exposed so a caller can, if desired, replace that with a
    data-calibrated value. Returns the fitted ``theta``.
    """
    from scipy.optimize import minimize_scalar

    diffs = np.asarray(elo_diffs, dtype=float)
    y = np.asarray(home_won, dtype=float)
    if diffs.size == 0:
        raise ValueError("no shootouts to calibrate theta on")

    def neg_ll(theta: float) -> float:
        z = theta * diffs
        # log-sigmoid via the stable log1p form; p = sigmoid(z).
        log_p = -np.logaddexp(0.0, -z)      # ln sigmoid(z)
        log_1mp = -np.logaddexp(0.0, z)     # ln (1 - sigmoid(z))
        return -float(np.sum(y * log_p + (1.0 - y) * log_1mp))

    res = minimize_scalar(
        neg_ll, bounds=theta_bounds, method="bounded", options={"xatol": 1e-9}
    )
    return float(res.x)


def _self_check() -> None:
    """Assert the two definition-of-done invariants hold (used by tests/main)."""
    # theta = 0 -> exact fair coin.
    assert penalty_prob(0.0, 2100.0, 1400.0) == 0.5

    # sum-to-1 and strong-vs-weak sanity on a representative tie.
    out = advance_prob(2.1, 0.7, -0.05, 2100.0, 1400.0, theta=0.0, kappa=0.8, K=10)
    total = out["p_home_advance"] + out["p_away_advance"]
    assert abs(total - 1.0) < 1e-9, f"advance probs must sum to 1, got {total}"
    assert out["p_home_advance"] > 0.5, "stronger side must advance > 50%"

    # degenerate lambda -> 0 must not NaN and still sum to 1.
    deg = advance_prob(0.0, 1.2, -0.05, 1500.0, 1500.0, theta=0.0)
    assert not math.isnan(deg["p_home_advance"])
    assert abs(deg["p_home_advance"] + deg["p_away_advance"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Entry point: python -m src.model.knockout
# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - illustrative CLI
    """Print the 90'/ET/penalty decomposition for a strong-vs-weak and even tie."""
    cfg = load_config()
    from src.model.lambdas import predict_lambdas

    _self_check()
    print("[ko] self-check OK: theta=0 fair coin, sums to 1, no NaN, strong>50%")

    # Use plausible fitted-ish coefficients so lambdas are realistic.
    c0, c1, rho = 0.2, 0.0018, -0.05

    def show(label, elo_h, elo_a):
        lam, mu = predict_lambdas(elo_h, elo_a, 0.0, c0, c1)
        P90 = score_matrix(lam, mu, rho, K=cfg.K_max)
        h, a, d = resolve_stage(P90)
        lam_et, mu_et = lam * _ET_MINUTES_RATIO * cfg.kappa, mu * _ET_MINUTES_RATIO * cfg.kappa
        eh, ea, el = resolve_stage(score_matrix(lam_et, mu_et, rho, K=cfg.K_max))
        out = advance_prob(lam, mu, rho, elo_h, elo_a, theta=cfg.theta, kappa=cfg.kappa, K=cfg.K_max)
        print(f"\n[ko] {label}: elo {elo_h:.0f} vs {elo_a:.0f}  (lambda {lam:.2f}/{mu:.2f})")
        print(f"     90' : home={h:.4f} away={a:.4f} draw={d:.4f}")
        print(f"     ET  : home={eh:.4f} away={ea:.4f} level={el:.4f}")
        print(f"     pen : home={penalty_prob(cfg.theta, elo_h, elo_a):.4f}")
        print(f"     ==> P(home advance)={out['p_home_advance']:.4f}  "
              f"P(away advance)={out['p_away_advance']:.4f}  "
              f"sum={out['p_home_advance'] + out['p_away_advance']:.12f}")

    show("strong vs weak", 2100.0, 1500.0)
    show("even matchup", 1800.0, 1800.0)


if __name__ == "__main__":  # pragma: no cover
    main()
