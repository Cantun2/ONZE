"""Elo differential -> expected goals lambda (spec §3.2).

This module owns the *mean* of the goals model: it maps each team's Elo rating
(as of the match date, leak-free) to the two expected-goal parameters
``(lambda_home, lambda_away)`` via a **global-coefficient Poisson regression**.

Model (spec §3.2), with an effective home bonus ``H_eff`` folded into the Elo
differential exactly as in §3.1 (``H_eff = H`` for a genuine host / non-neutral
venue, ``0`` on neutral ground)::

    log lambda_home = c0 + c1 * (R_home + H_eff - R_away)
    log lambda_away = c0 + c1 * (R_away - R_home - H_eff)

Only THREE parameters (``c0``, ``c1`` — plus ``rho`` in dixon_coles) are fit,
over tens of thousands of matches, so the estimate is very stable (spec §3.2).

``c0`` sets the average scoring level; ``c1 > 0`` the sensitivity to the force
differential. Note the differential is *anti-symmetric* between the two sides:
the home side sees ``+(R_home + H_eff - R_away)``, the away side its negation.

Design decisions
----------------
* Ratings are read **as of the match date** from ``ratings_history`` (each row
  is a team's PRE-match rating, spec §3.1/§3.6) — never post-match, so fitting
  is leak-free. We consume :func:`src.ratings.elo.rating_as_of`; we do NOT
  recompute Elo here (that is the ratings agent's job).
* ``H`` is passed in (from ``config.load_config().H``), never hard-coded.
* The regression is a standard Poisson GLM with a log link. We stack the two
  sides (home, away) of every match into a single design matrix with one shared
  intercept ``c0`` and one shared slope ``c1`` on the (signed) differential, and
  maximise the Poisson log-likelihood of the observed goals. We use
  ``statsmodels`` GLM when available (closed-form IRLS) and fall back to a
  hand-rolled scipy Newton fit otherwise; both maximise the same likelihood.

Public API
----------
``build_design``        (differential, goals) rows for every played match.
``fit_lambda_coeffs``   MLE Poisson regression -> (c0, c1) [+ loglik via _fit_full].
``predict_lambdas``     (elo_home, elo_away, H_eff, c0, c1) -> (lam, mu).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import load_config


# ---------------------------------------------------------------------------
# Prediction (spec §3.2)
# ---------------------------------------------------------------------------
def predict_lambdas(
    elo_home: float,
    elo_away: float,
    H_eff: float,
    c0: float,
    c1: float,
) -> tuple[float, float]:
    """Expected goals ``(lambda_home, lambda_away)`` from Elo (spec §3.2).

    ``H_eff`` is the *effective* home bonus already decided by the caller
    (``H`` for a host / non-neutral venue, ``0`` on neutral ground) — this
    function does not re-derive it. The differential is anti-symmetric::

        diff_home = R_home + H_eff - R_away
        diff_away = R_away - R_home - H_eff   (= -diff_home)
        lambda_home = exp(c0 + c1 * diff_home)
        lambda_away = exp(c0 + c1 * diff_away)

    Returns strictly positive floats.
    """
    diff_home = (elo_home + H_eff) - elo_away
    lam = math.exp(c0 + c1 * diff_home)
    mu = math.exp(c0 + c1 * (-diff_home))
    return lam, mu


# ---------------------------------------------------------------------------
# Design matrix: one (differential, goals) row per side per played match
# ---------------------------------------------------------------------------
def _load_ratings_history(con) -> pd.DataFrame:
    """Read the full ``ratings_history`` frame (team_id, date, elo) from DuckDB."""
    df = con.execute(
        "SELECT team_id, date, elo FROM ratings_history"
    ).fetchdf()
    return df


def _prematch_elos_by_sequence(
    m: pd.DataFrame, rh: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Home/away PRE-match elos for every match via a row-exact sequence join.

    The Elo engine (:func:`src.ratings.elo.compute_elo_history`) walks matches in
    ``(date, match_id)`` order and, for every played match, emits its two
    pre-match ``ratings_history`` rows in that order — **home row then away row**.
    So if we enumerate each team's *appearances* (home OR away, one per match it
    played) in ``(date, match_id, home-before-away)`` order, a team's *k*-th
    appearance's pre-match rating is its *k*-th ``ratings_history`` row (emit
    order). We therefore key the join on ``(team_id, appearance-seq)`` — **not**
    on calendar date.

    This is immune to the same-day double-header ambiguity that a date-only
    ``merge_asof`` has: two matches on one day become two distinct appearances
    with distinct sequence indices, so each gets its OWN pre-match rating and can
    never inherit its sibling's post-result rating (§3.6, no self-leak).

    Contract: ``m`` must be a time-PREFIX of the full played history (the full
    frame and every ``fit_up_to`` window are, since ``_played_before`` splits on
    a date with no partial same-day), so a team's appearance index within ``m``
    equals its index in the full history and lines up with ``rh``. ``rh`` must be
    in emit order for same-date ties (true for both sources: the in-memory frame
    from ``compute_elo_history`` and a single ordered DB insert); the caller
    stable-sorts it by ``date`` only, preserving within-day emit order.

    Returns ``(r_home, r_away)`` arrays aligned to ``m``'s row order.
    """
    ordered = m.sort_values(["date", "match_id"], kind="mergesort")
    n = len(ordered)

    # Melt to one row per APPEARANCE, home before away within each match, so the
    # long order is exactly the Elo engine's emit order.
    appear = pd.DataFrame(
        {
            "team_id": np.concatenate(
                [ordered["home_id"].to_numpy(), ordered["away_id"].to_numpy()]
            ),
            # side key: 0 = home, 1 = away — used only to order within a match.
            "side": np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)]),
            "match_pos": np.concatenate([np.arange(n), np.arange(n)]),
        }
    ).sort_values(["match_pos", "side"], kind="mergesort")
    appear["seq"] = appear.groupby("team_id").cumcount()

    rh_seq = rh.copy()
    rh_seq["seq"] = rh_seq.groupby("team_id").cumcount()
    appear = appear.merge(
        rh_seq[["team_id", "seq", "elo"]], on=["team_id", "seq"], how="left"
    )

    # Split back into home / away, in ordered-match order, then map to m's order.
    home_elo = appear.loc[appear["side"] == 0].sort_values("match_pos")["elo"].to_numpy()
    away_elo = appear.loc[appear["side"] == 1].sort_values("match_pos")["elo"].to_numpy()

    # ordered rows -> original m order
    order_index = ordered.index.to_numpy()
    r_home = pd.Series(home_elo, index=order_index).reindex(m.index).to_numpy(dtype=float)
    r_away = pd.Series(away_elo, index=order_index).reindex(m.index).to_numpy(dtype=float)
    return r_home, r_away


def match_features(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    H: float,
    con=None,
) -> pd.DataFrame:
    """Per-match features via a leak-free, row-exact as-of Elo join (training path).

    For every played match this returns one row with:

    * ``diff_home``  the home-side Elo differential ``R_home + H_eff - R_away``,
    * ``x``, ``y``   home / away goals,
    * ``date``       the match date.

    **As-of semantics (training).** Each team's PRE-match rating for a match is
    the ``ratings_history`` row the Elo engine recorded *for that match* (spec
    §3.1: history stores the rating *before* the match). We attach it by a
    row-exact ``(team_id, appearance-sequence)`` join — see
    :func:`_prematch_elos_by_sequence` — NOT a calendar-date merge. This is
    strictly pre-match (never post-match), and, unlike a date-only join, is
    immune to same-day double-headers: a team that plays twice on one day gets
    each match's own pre-match rating, with no sibling-match contamination
    (§3.6, absolute no-leak rule).

    (Contrast :func:`src.ratings.elo.rating_as_of`, whose strict ``<`` is the
    right tool for a *future fixture* whose own pre-match row is not yet in the
    table; for historical training rows the recorded on-date row IS the pre-match
    rating, which the sequence join selects exactly.)

    Supply either a DuckDB ``con`` or an in-memory ``ratings_history`` frame.
    ``matches`` must be a time-prefix of the played history (the full frame and
    every ``fit_up_to`` window are); see :func:`_prematch_elos_by_sequence`.
    """
    if con is None and ratings_history is None:
        raise ValueError("match_features needs either a `con` or a `ratings_history` frame")
    if ratings_history is None:
        ratings_history = _load_ratings_history(con)

    m = matches.copy()
    # Only played matches are training observations.
    m = m[m["home_goals"].notna() & m["away_goals"].notna()].copy()
    if m.empty:
        return pd.DataFrame(columns=["diff_home", "x", "y", "date"])

    m["date"] = pd.to_datetime(m["date"])
    has_host = "host_flag" in m.columns
    if has_host:
        h_eff = np.where(
            m["host_flag"].astype(bool),
            H,
            np.where(m["neutral"].astype(bool), 0.0, H),
        )
    else:
        h_eff = np.where(m["neutral"].astype(bool), 0.0, H)
    m["h_eff"] = h_eff.astype(float)

    # History in emit order for same-date ties: stable-sort by date only, which
    # preserves the within-day (home-then-away, match-order) emit order of both
    # our sources (compute_elo_history frame / single ordered DB insert).
    rh = ratings_history.copy()
    rh["date"] = pd.to_datetime(rh["date"])
    rh = rh.sort_values("date", kind="mergesort").reset_index(drop=True)

    r_home, r_away = _prematch_elos_by_sequence(m, rh)

    # Teams with no prior history row fall back to the base rating (debut).
    base = load_config().base_rating
    r_home = np.where(np.isnan(r_home), base, r_home)
    r_away = np.where(np.isnan(r_away), base, r_away)

    diff_home = (r_home + m["h_eff"].to_numpy(dtype=float)) - r_away
    return pd.DataFrame(
        {
            "diff_home": diff_home,
            "x": m["home_goals"].to_numpy(dtype=float),
            "y": m["away_goals"].to_numpy(dtype=float),
            "date": m["date"].dt.date.to_numpy(),
        }
    )


def build_design(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    H: float,
    con=None,
) -> pd.DataFrame:
    """Stack both sides of every played match into a Poisson-regression design.

    For each played match we emit **two** rows — the home side and the away
    side — each with:

    * ``diff``   the signed Elo differential seen by that side
                 (home: ``R_home + H_eff - R_away``; away: its negation),
    * ``goals``  the goals that side scored,
    * ``date``   the match date (kept for time-decay weighting downstream).

    Ratings are the leak-free as-of pre-match ratings from :func:`match_features`
    (vectorised join; see its docstring for the as-of semantics). Supply either a
    DuckDB ``con`` or an in-memory ``ratings_history`` frame.
    """
    feats = match_features(matches, ratings_history, H, con=con)
    if feats.empty:
        return pd.DataFrame({"diff": [], "goals": [], "date": []})

    diff = feats["diff_home"].to_numpy(dtype=float)
    # Interleave home (+diff, x) and away (-diff, y) rows.
    diffs = np.empty(2 * len(feats))
    diffs[0::2] = diff
    diffs[1::2] = -diff
    goals = np.empty(2 * len(feats))
    goals[0::2] = feats["x"].to_numpy()
    goals[1::2] = feats["y"].to_numpy()
    dates = np.repeat(feats["date"].to_numpy(), 2)
    return pd.DataFrame({"diff": diffs, "goals": goals, "date": dates})


# ---------------------------------------------------------------------------
# Poisson MLE (spec §3.2) — statsmodels IRLS with a scipy fallback
# ---------------------------------------------------------------------------
class LambdaFit(NamedTuple):
    """Result of the lambda-coefficient fit."""

    c0: float
    c1: float
    loglik: float
    n_obs: int


def _poisson_loglik(c0: float, c1: float, diff: np.ndarray, goals: np.ndarray) -> float:
    """Full Poisson log-likelihood (incl. -log(y!) constant) of the design."""
    eta = c0 + c1 * diff
    lam = np.exp(eta)
    # sum_i [ y_i * eta_i - lam_i - log(y_i!) ]
    from scipy.special import gammaln

    return float(np.sum(goals * eta - lam - gammaln(goals + 1.0)))


def _fit_full(design: pd.DataFrame) -> LambdaFit:
    """Fit (c0, c1) by Poisson MLE on a design frame; return coeffs + loglik."""
    diff = design["diff"].to_numpy(dtype=float)
    goals = design["goals"].to_numpy(dtype=float)
    n = len(goals)
    if n == 0:
        raise ValueError("cannot fit lambda coefficients on an empty design")

    try:
        import statsmodels.api as sm

        X = sm.add_constant(diff, has_constant="add")  # [1, diff]
        model = sm.GLM(goals, X, family=sm.families.Poisson())
        res = model.fit()
        c0, c1 = float(res.params[0]), float(res.params[1])
    except Exception:
        # Fallback: Newton-Raphson on the 2-parameter Poisson score.
        c0, c1 = _fit_scipy(diff, goals)

    loglik = _poisson_loglik(c0, c1, diff, goals)
    return LambdaFit(c0=c0, c1=c1, loglik=loglik, n_obs=n)


def _fit_scipy(diff: np.ndarray, goals: np.ndarray) -> tuple[float, float]:
    """Minimise the negative Poisson log-likelihood in (c0, c1) with scipy."""
    from scipy.optimize import minimize

    def neg_ll(theta: np.ndarray) -> float:
        eta = theta[0] + theta[1] * diff
        # guard against overflow for extreme differentials
        eta = np.clip(eta, -30.0, 30.0)
        lam = np.exp(eta)
        return float(np.sum(lam - goals * eta))

    def grad(theta: np.ndarray) -> np.ndarray:
        eta = np.clip(theta[0] + theta[1] * diff, -30.0, 30.0)
        lam = np.exp(eta)
        r = lam - goals
        return np.array([r.sum(), (r * diff).sum()])

    x0 = np.array([math.log(max(goals.mean(), 1e-3)), 0.0])
    res = minimize(neg_ll, x0, jac=grad, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def fit_lambda_coeffs(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    H: float,
    con=None,
) -> tuple[float, float]:
    """MLE Poisson regression of log-lambda on the Elo differential (spec §3.2).

    Returns ``(c0, c1)``. Ratings are consumed **as of each match's date**
    (leak-free) from either a DuckDB ``con`` or an in-memory ``ratings_history``
    frame. ``H`` is the home bonus from config; ``H_eff`` per row is derived from
    ``neutral`` / ``host_flag`` exactly as the Elo engine does (spec §3.1).

    For the achieved log-likelihood and observation count, use :func:`_fit_full`
    (or the ``main`` in dixon_coles which reports them).
    """
    design = build_design(matches, ratings_history, H, con=con)
    fit = _fit_full(design)
    return fit.c0, fit.c1
