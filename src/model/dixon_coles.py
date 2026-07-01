"""Dixon-Coles goals model: tau correction, time-weighted MLE, score matrix.

This is the mathematical core of ONZE (spec §3.3). Given the per-match expected
goals ``(lambda, mu)`` from :mod:`src.model.lambdas` (§3.2), it builds the joint
score distribution ``P(X=x, Y=y)`` with the Dixon-Coles low-score correction and
fits the single dependence parameter ``rho`` by time-weighted maximum
likelihood.

The four tau cases (spec §3.3), reproduced EXACTLY
--------------------------------------------------
::

    tau(x, y) = 1 - lambda*mu*rho   if (x, y) == (0, 0)
                1 + lambda*rho       if (x, y) == (0, 1)
                1 + mu*rho           if (x, y) == (1, 0)
                1 - rho              if (x, y) == (1, 1)
                1                    otherwise

Joint law (spec §3.3), the boxed formula::

    P(X=x, Y=y) = tau(x, y) * Pois(x; lambda) * Pois(y; mu)

Time-weighted log-likelihood (spec §3.3), reproduced term-for-term::

    l(c0, c1, rho) = sum_m phi(dt_m) * [ ln tau(x_m, y_m)
                                         - lambda_m + x_m * ln lambda_m
                                         - mu_m     + y_m * ln mu_m ]

    phi(dt) = exp(-xi * dt),   dt_m = (t_ref - date_m) in DAYS

where ``t_ref`` is the most recent match date in the fit window (so the newest
match has weight 1 and older ones decay). Factorial constants ``-ln(x!)`` are
dropped exactly as the spec allows (they do not depend on the parameters). The
``lambda_m, mu_m`` come from :func:`src.model.lambdas.predict_lambdas` on the
as-of Elo, so ``(c0, c1, rho)`` can be fit jointly and leak-free.

Validity of rho (spec §3.3, §7.7)
---------------------------------
tau must stay >= 0 on all four low scores. The binding constraints are
``1 - lambda*mu*rho >= 0`` (for rho > 0) and ``1 - rho >= 0`` / ``1 + lambda*rho
>= 0`` (for rho < 0). Over realistic low-scoring lambdas the spec's region is
``rho in [-0.2, 0]``, which we adopt as the optimisation bound (see
``RHO_BOUNDS``). Empirically rho is small and negative (independent-Poisson
under-weights draws), so this bound is not binding at the optimum.

Public API
----------
``tau``             the four-case low-score factor.
``score_matrix``    (K+1)x(K+1) renormalised joint pmf.
``fit_rho``         time-weighted MLE of rho given fixed (c0, c1).
``fit_up_to``       leak-free joint fit of (c0, c1, rho) on matches < date.
``fit_full_history``joint fit on all played matches.
``main``            ``python -m src.model.dixon_coles`` end-to-end + persistence.
"""

from __future__ import annotations

import datetime as _dt
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.config import Config, load_config
from src.model.lambdas import _fit_full, match_features, predict_lambdas


# rho region keeping tau >= 0 on all low scores for realistic lambdas (spec §3.3).
RHO_BOUNDS = (-0.2, 0.0)


# ---------------------------------------------------------------------------
# tau: the Dixon-Coles low-score correction (spec §3.3) — verbatim four cases
# ---------------------------------------------------------------------------
def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score factor tau(x, y) (spec §3.3), the four cases:

    * ``(0, 0)`` -> ``1 - lam*mu*rho``
    * ``(0, 1)`` -> ``1 + lam*rho``
    * ``(1, 0)`` -> ``1 + mu*rho``
    * ``(1, 1)`` -> ``1 - rho``
    * otherwise  -> ``1``

    ``lam`` = lambda_home, ``mu`` = lambda_away. With ``rho = 0`` every case
    collapses to ``1`` (independent Poisson).
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Score matrix (spec §3.3 boxed law + §7.5 truncate-then-renormalise)
# ---------------------------------------------------------------------------
def _poisson_pmf_vector(rate: float, K: int) -> np.ndarray:
    """Poisson pmf over counts 0..K, numerically safe for rate -> 0.

    Uses the log-pmf ``k*ln(rate) - rate - ln(k!)`` so a small ``rate`` never
    produces NaN. For ``rate == 0`` the mass is all at k=0 (limit of Poisson).
    """
    k = np.arange(K + 1)
    if rate <= 0.0:
        pmf = np.zeros(K + 1)
        pmf[0] = 1.0
        return pmf
    log_pmf = k * np.log(rate) - rate - gammaln(k + 1.0)
    return np.exp(log_pmf)


def score_matrix(lam: float, mu: float, rho: float, K: int = 10) -> np.ndarray:
    """Renormalised joint pmf ``P(x, y)`` for ``x, y in {0..K}`` (spec §3.3).

    Entry ``[x, y] = tau(x, y, lam, mu, rho) * Pois(x; lam) * Pois(y; mu)``,
    then the whole matrix is **renormalised** to sum to 1 after truncation at
    ``K`` and the tau correction (spec §7.5). With ``rho == 0`` the tau block is
    all ones, so the result is exactly ``outer(Pois(lam), Pois(mu))`` truncated
    and renormalised — i.e. independent Poisson (self-checked in ``main``).

    Numerically safe when ``lam`` or ``mu`` -> 0 (mass collapses onto the 0-goal
    row/column; no NaN).
    """
    px = _poisson_pmf_vector(lam, K)
    py = _poisson_pmf_vector(mu, K)
    P = np.outer(px, py)  # independent-Poisson block, [x, y]

    # Apply tau only on the four low-score cells (everything else multiplies by 1).
    P[0, 0] *= (1.0 - lam * mu * rho)
    if K >= 1:
        P[0, 1] *= (1.0 + lam * rho)
        P[1, 0] *= (1.0 + mu * rho)
        P[1, 1] *= (1.0 - rho)

    # tau can dip a corrected cell slightly negative only if rho is out of the
    # valid region; clip defensively so the matrix stays a proper pmf.
    np.clip(P, 0.0, None, out=P)

    total = P.sum()
    if total <= 0.0:  # pragma: no cover - defensive; cannot happen for valid inputs
        # degenerate: fall back to a point mass at (0, 0)
        P[:] = 0.0
        P[0, 0] = 1.0
        return P
    P /= total  # renormalise after truncation + tau (spec §7.5)
    return P


# ---------------------------------------------------------------------------
# Time-weighted Dixon-Coles log-likelihood (spec §3.3, term-for-term)
# ---------------------------------------------------------------------------
class _MatchTerms(NamedTuple):
    """Pre-computed per-match arrays for the rho likelihood (lambda fixed)."""

    lam: np.ndarray   # lambda_home per match
    mu: np.ndarray    # lambda_away per match
    x: np.ndarray     # home goals
    y: np.ndarray     # away goals
    weight: np.ndarray  # phi(dt) time-decay weight


def _terms_from_features(feats: pd.DataFrame, c0: float, c1: float, xi: float) -> _MatchTerms:
    """Turn per-match features into (lambda, mu, x, y, weight) for the rho likelihood.

    lambda/mu come from the log-linear map (spec §3.2) on the as-of Elo
    differential (leak-free, from :func:`match_features`). Weights are
    ``phi(dt) = exp(-xi * dt)`` with ``dt`` = days back from the most recent
    match in the frame (spec §3.3), so the newest match has weight 1.
    """
    if feats.empty:
        raise ValueError("no played matches to fit rho on")

    diff = feats["diff_home"].to_numpy(dtype=float)
    lam = np.exp(c0 + c1 * diff)          # lambda_home
    mu = np.exp(c0 + c1 * (-diff))        # lambda_away (anti-symmetric)
    x = feats["x"].to_numpy(dtype=float)
    y = feats["y"].to_numpy(dtype=float)

    dates = pd.to_datetime(feats["date"])
    t_ref = dates.max()                   # most recent match -> weight 1
    dt_days = (t_ref - dates).dt.days.to_numpy(dtype=float)
    weight = np.exp(-xi * dt_days)

    return _MatchTerms(lam=lam, mu=mu, x=x, y=y, weight=weight)


def _lambda_terms_from_design(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    con,
    H: float,
    c0: float,
    c1: float,
    xi: float,
) -> _MatchTerms:
    """Vectorised per-match (lambda, mu, x, y, weight) for the rho likelihood.

    Thin wrapper: builds leak-free as-of features via :func:`match_features`
    then delegates to :func:`_terms_from_features`.
    """
    feats = match_features(matches, ratings_history, H, con=con)
    return _terms_from_features(feats, c0, c1, xi)


def _ln_tau_vector(terms: _MatchTerms, rho: float) -> np.ndarray:
    """Vector of ``ln tau(x_m, y_m, lam_m, mu_m, rho)`` over all matches.

    Vectorised form of the four-case :func:`tau`, matching it cell-for-cell.
    """
    lam, mu, x, y = terms.lam, terms.mu, terms.x, terms.y
    t = np.ones_like(lam)  # default tau = 1 (the "otherwise" case)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho

    # tau must stay > 0 inside the log; the RHO_BOUNDS keep it so, but clip to a
    # tiny floor as a defensive guard against a boundary evaluation.
    np.clip(t, 1e-12, None, out=t)
    return np.log(t)


def _weighted_loglik(terms: _MatchTerms, rho: float) -> float:
    """Time-weighted Dixon-Coles log-likelihood (spec §3.3), term-for-term.

    sum_m phi(dt_m) * [ ln tau_m - lam_m + x_m ln lam_m - mu_m + y_m ln mu_m ].
    Factorial constants are dropped (independent of the parameters, spec §3.3).
    """
    lam, mu, x, y, w = terms.lam, terms.mu, terms.x, terms.y, terms.weight
    ln_tau = _ln_tau_vector(terms, rho)
    per_match = ln_tau - lam + x * np.log(lam) - mu + y * np.log(mu)
    return float(np.sum(w * per_match))


def fit_rho(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    coeffs: tuple[float, float],
    xi: float,
    con=None,
    H: float | None = None,
) -> float:
    """Time-weighted MLE of the dependence parameter ``rho`` (spec §3.3).

    Given fixed lambda coefficients ``coeffs = (c0, c1)``, maximise the
    time-weighted Dixon-Coles log-likelihood over ``rho`` (equivalently minimise
    ``-l``), bounded to :data:`RHO_BOUNDS` so tau stays >= 0 on all low scores.

    lambda/mu per match come from the as-of Elo (leak-free). ``xi`` is the decay
    rate (passed in, never hard-coded). Uses a bounded scalar minimiser — the
    1-D problem is smooth and the interval tiny, so this is both robust and
    equivalent to L-BFGS-B on a single coordinate.
    """
    if H is None:
        H = load_config().H
    c0, c1 = coeffs
    terms = _lambda_terms_from_design(matches, ratings_history, con, H, c0, c1, xi)

    def neg_ll(rho: float) -> float:
        return -_weighted_loglik(terms, rho)

    res = minimize_scalar(
        neg_ll,
        bounds=RHO_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(res.x)


# ---------------------------------------------------------------------------
# Joint fit of (c0, c1, rho) — leak-free windowing (spec §3.3, §3.6)
# ---------------------------------------------------------------------------
class ModelFit(NamedTuple):
    """Fitted goal-model coefficients + achieved (weighted) log-likelihood."""

    c0: float
    c1: float
    rho: float
    loglik: float          # time-weighted DC log-likelihood at (c0, c1, rho)
    lambda_loglik: float   # Poisson loglik of the lambda GLM (for reference)
    n_matches: int
    xi: float


def _fit_coeffs_and_rho(
    matches: pd.DataFrame,
    ratings_history: pd.DataFrame | None,
    con,
    H: float,
    xi: float,
) -> ModelFit:
    """Fit (c0, c1) by Poisson GLM, then rho by time-weighted MLE; report loglik.

    We estimate (c0, c1) and rho in sequence (the spec's joint objective is
    dominated by the Poisson means; rho is a small low-score refinement). Both
    stages use only the ``matches`` passed in and their as-of Elo, so any
    windowing applied by the caller (e.g. ``fit_up_to``) is respected.
    """
    # One vectorised as-of join, reused for both the lambda GLM and the rho MLE.
    feats = match_features(matches, ratings_history, H, con=con)
    if feats.empty:
        raise ValueError("no played matches in the fit window")

    # Stack both sides of every match into the Poisson-GLM design.
    diff = feats["diff_home"].to_numpy(dtype=float)
    diffs = np.empty(2 * len(feats))
    diffs[0::2] = diff
    diffs[1::2] = -diff
    goals = np.empty(2 * len(feats))
    goals[0::2] = feats["x"].to_numpy()
    goals[1::2] = feats["y"].to_numpy()
    lam_fit = _fit_full(pd.DataFrame({"diff": diffs, "goals": goals}))

    # rho by time-weighted MLE, reusing the same features.
    terms = _terms_from_features(feats, lam_fit.c0, lam_fit.c1, xi)

    def neg_ll(rho: float) -> float:
        return -_weighted_loglik(terms, rho)

    res = minimize_scalar(
        neg_ll, bounds=RHO_BOUNDS, method="bounded", options={"xatol": 1e-6}
    )
    rho = float(res.x)
    loglik = _weighted_loglik(terms, rho)
    return ModelFit(
        c0=lam_fit.c0,
        c1=lam_fit.c1,
        rho=rho,
        loglik=loglik,
        lambda_loglik=lam_fit.loglik,
        n_matches=len(terms.lam),
        xi=xi,
    )


def _played_before(matches: pd.DataFrame, date) -> pd.DataFrame:
    """Rows strictly before ``date`` (leak-free window for a backtest at ``date``)."""
    if isinstance(date, str):
        date = _dt.date.fromisoformat(date[:10])
    elif isinstance(date, _dt.datetime):
        date = date.date()
    d = pd.to_datetime(matches["date"]).dt.date
    return matches.loc[d < date].copy()


def fit_up_to(
    date,
    matches: pd.DataFrame | None = None,
    ratings_history: pd.DataFrame | None = None,
    con=None,
    xi: float | None = None,
    H: float | None = None,
) -> ModelFit:
    """Fit ``(c0, c1, rho)`` using ONLY matches strictly before ``date`` (spec §3.6).

    This is the leak-free entry point for eval-engineer's walk-forward backtest:
    to predict a fixture on ``date`` you call ``fit_up_to(date)`` and every
    training match is guaranteed earlier than ``date``. The as-of Elo lookup
    (``rating_as_of``) is itself strictly-before, so ratings are leak-free too.

    ``matches`` / ``ratings_history`` may be supplied in-memory, or omitted to
    read from the DuckDB ``con`` (or the configured store). ``xi`` / ``H``
    default to config.
    """
    cfg = load_config()
    if xi is None:
        xi = cfg.xi
    if H is None:
        H = cfg.H

    own_con = False
    if matches is None:
        import duckdb

        if con is None:
            con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
            own_con = True
        matches = con.execute(
            "SELECT match_id, date, home_id, away_id, home_goals, away_goals, "
            "neutral, tournament, importance_k FROM matches "
            "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"
        ).fetchdf()

    try:
        window = _played_before(matches, date)
        if window.empty:
            raise ValueError(f"no matches strictly before {date} to fit on")
        return _fit_coeffs_and_rho(window, ratings_history, con, H, xi)
    finally:
        if own_con:
            con.close()


def fit_full_history(
    matches: pd.DataFrame | None = None,
    ratings_history: pd.DataFrame | None = None,
    con=None,
    xi: float | None = None,
    H: float | None = None,
) -> ModelFit:
    """Fit ``(c0, c1, rho)`` on ALL played matches (production coefficients)."""
    cfg = load_config()
    if xi is None:
        xi = cfg.xi
    if H is None:
        H = cfg.H

    own_con = False
    if matches is None:
        import duckdb

        if con is None:
            con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
            own_con = True
        matches = con.execute(
            "SELECT match_id, date, home_id, away_id, home_goals, away_goals, "
            "neutral, tournament, importance_k FROM matches "
            "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"
        ).fetchdf()

    try:
        return _fit_coeffs_and_rho(matches, ratings_history, con, H, xi)
    finally:
        if own_con:
            con.close()


# ---------------------------------------------------------------------------
# Self-checks (asserted in main and in tests)
# ---------------------------------------------------------------------------
def _self_check_score_matrix() -> None:
    """Assert score_matrix sums to 1 and reduces to independent Poisson at rho=0."""
    lam, mu, K = 1.8, 1.1, 10
    P = score_matrix(lam, mu, 0.0, K=K)
    assert abs(P.sum() - 1.0) < 1e-12, "score_matrix must sum to 1"

    # rho = 0 -> exactly the (truncated, renormalised) independent-Poisson outer.
    px = _poisson_pmf_vector(lam, K)
    py = _poisson_pmf_vector(mu, K)
    indep = np.outer(px, py)
    indep /= indep.sum()
    assert np.allclose(P, indep, atol=1e-12), "rho=0 must reduce to independent Poisson"

    # a valid negative rho still sums to 1 and stays non-negative
    P2 = score_matrix(lam, mu, -0.05, K=K)
    assert abs(P2.sum() - 1.0) < 1e-12
    assert (P2 >= 0).all()

    # lambda -> 0 must not NaN and must put all mass on the 0-goal row
    P0 = score_matrix(0.0, mu, -0.05, K=K)
    assert not np.isnan(P0).any()
    assert abs(P0.sum() - 1.0) < 1e-12
    assert abs(P0[1:, :].sum()) < 1e-12, "lambda=0 -> no home goals"


# ---------------------------------------------------------------------------
# Persistence of fitted coefficients (for downstream predict/eval)
# ---------------------------------------------------------------------------
_COEFFS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS model_coeffs (
    model_version VARCHAR,
    c0            DOUBLE,
    c1            DOUBLE,
    rho           DOUBLE,
    xi            DOUBLE,
    loglik        DOUBLE,
    n_matches     BIGINT,
    fit_up_to     DATE,        -- NULL = full history
    created_at    TIMESTAMP
);
"""


def persist_coeffs(
    con,
    fit: ModelFit,
    model_version: str = "dc-v1",
    fit_up_to_date: _dt.date | None = None,
) -> None:
    """Persist a fitted ``ModelFit`` into a ``model_coeffs`` table (idempotent).

    Downstream predict/eval read the latest row for a given ``model_version``.
    ``fit_up_to_date`` records the leak-free window boundary (NULL = full history).
    """
    con.execute(_COEFFS_TABLE_DDL)
    con.execute(
        "DELETE FROM model_coeffs WHERE model_version = ?", [model_version]
    )
    con.execute(
        "INSERT INTO model_coeffs VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())",
        [
            model_version,
            fit.c0,
            fit.c1,
            fit.rho,
            fit.xi,
            fit.loglik,
            fit.n_matches,
            fit_up_to_date,
        ],
    )


# ---------------------------------------------------------------------------
# Entry point: python -m src.model.dixon_coles
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    """Fit (c0, c1, rho) on the full history, print them + loglik, run self-checks.

    Also prints example lambdas for a strong-vs-weak matchup and the most-likely
    score, and (best-effort) persists the coefficients to ``model_coeffs``.
    """
    import duckdb

    cfg: Config = load_config()

    # Self-checks first: these are cheap and must always hold.
    _self_check_score_matrix()
    print("[dc] self-check OK: score_matrix sums to 1 & reduces to indep. Poisson at rho=0")

    con = duckdb.connect(str(cfg.paths.db_path))
    try:
        fit = fit_full_history(con=con, xi=cfg.xi, H=cfg.H)
        print("\n[dc] Fitted goal-model coefficients (full history, spec §3.2/§3.3):")
        print(f"     c0  = {fit.c0:.6f}")
        print(f"     c1  = {fit.c1:.8f}")
        print(f"     rho = {fit.rho:.6f}   (bounds {RHO_BOUNDS})")
        print(f"     xi  = {fit.xi:.8g}  (half-life ~= {cfg.xi_half_life_days:.0f} days)")
        print(f"     time-weighted DC log-likelihood = {fit.loglik:,.2f}")
        print(f"     (reference Poisson-GLM log-likelihood = {fit.lambda_loglik:,.2f})")
        print(f"     matches used = {fit.n_matches:,}")

        # Example: a strong home team vs a weak away team (neutral World Cup venue).
        elo_strong, elo_weak, H_eff = 2050.0, 1550.0, 0.0
        lam, mu = predict_lambdas(elo_strong, elo_weak, H_eff, fit.c0, fit.c1)
        P = score_matrix(lam, mu, fit.rho, K=cfg.K_max)
        x_ml, y_ml = np.unravel_index(int(np.argmax(P)), P.shape)
        # 1N2 for context: home win = x>y (strictly-lower triangle), away = x<y.
        p_home = float(np.tril(P, -1).sum())
        p_draw = float(np.trace(P))
        p_away = float(np.triu(P, 1).sum())
        print("\n[dc] Example strong(2050) vs weak(1550), neutral venue:")
        print(f"     (lambda_home, lambda_away) = ({lam:.3f}, {mu:.3f})")
        print(f"     most-likely score = {x_ml}-{y_ml}  (P = {P[x_ml, y_ml]:.4f})")
        print(f"     P(home)={p_home:.3f}  P(draw)={p_draw:.3f}  P(away)={p_away:.3f}")
        print(f"     score_matrix sum = {P.sum():.12f}")

        # A second, closer example for sanity.
        lam2, mu2 = predict_lambdas(1900.0, 1850.0, cfg.H, fit.c0, fit.c1)
        P2 = score_matrix(lam2, mu2, fit.rho, K=cfg.K_max)
        x2, y2 = np.unravel_index(int(np.argmax(P2)), P2.shape)
        print("\n[dc] Example home(1900,+H) vs away(1850):")
        print(f"     (lambda_home, lambda_away) = ({lam2:.3f}, {mu2:.3f})")
        print(f"     most-likely score = {x2}-{y2}  (P = {P2[x2, y2]:.4f})")

        # Persist for downstream predict/eval.
        try:
            persist_coeffs(con, fit, model_version="dc-v1", fit_up_to_date=None)
            con.commit()
            print("\n[dc] coefficients persisted to model_coeffs (model_version='dc-v1')")
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            print(f"\n[dc] (warning) could not persist coefficients: {exc}")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    main()
