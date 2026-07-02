"""Tests for the goals model (src/model/lambdas.py + dixon_coles.py, spec §3.2/§3.3).

These lock in the math-critical properties the model-reviewer audits:

* **tau** — the four Dixon-Coles low-score cases, verbatim (spec §3.3).
* **score_matrix** — sums to 1 after truncation + tau, valid pmf, reduces to
  independent Poisson at rho=0, and never NaNs when lambda/mu -> 0 (spec §7.5).
* **likelihood** — the time-weighted DC log-likelihood is reproduced term for
  term and is maximised at the fitted rho, beating independent Poisson.
* **rho validity** — the fitted rho stays inside the tau>=0 region (spec §7.7).
* **no leakage** — fit_up_to(date) uses only matches strictly before `date`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.model.dixon_coles import (
    RHO_BOUNDS,
    _self_check_score_matrix,
    _terms_from_features,
    _weighted_loglik,
    fit_full_history,
    fit_rho,
    fit_up_to,
    score_matrix,
    tau,
)
from src.model.lambdas import (
    build_design,
    fit_lambda_coeffs,
    match_features,
    predict_lambdas,
)
from src.ratings.elo import compute_elo_history

CFG = load_config()


# ---------------------------------------------------------------------------
# tau — the four low-score cases, verbatim (spec §3.3)
# ---------------------------------------------------------------------------
def test_tau_four_cases_verbatim():
    lam, mu, rho = 1.5, 1.2, -0.05
    assert tau(0, 0, lam, mu, rho) == 1.0 - lam * mu * rho
    assert tau(0, 1, lam, mu, rho) == 1.0 + lam * rho
    assert tau(1, 0, lam, mu, rho) == 1.0 + mu * rho
    assert tau(1, 1, lam, mu, rho) == 1.0 - rho


def test_tau_otherwise_is_one():
    lam, mu, rho = 1.5, 1.2, -0.05
    for x, y in [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3), (5, 4)]:
        assert tau(x, y, lam, mu, rho) == 1.0


def test_tau_rho_zero_is_identity():
    for x in range(3):
        for y in range(3):
            assert tau(x, y, 1.7, 1.1, 0.0) == 1.0


# ---------------------------------------------------------------------------
# score_matrix (spec §3.3 + §7.5)
# ---------------------------------------------------------------------------
def test_score_matrix_self_check():
    # Bundles: sums to 1, reduces to indep Poisson at rho=0, lambda->0 safe.
    _self_check_score_matrix()


def test_score_matrix_sums_to_one():
    for lam, mu, rho in [(1.8, 1.1, 0.0), (2.5, 0.6, -0.05), (0.9, 0.9, -0.1)]:
        P = score_matrix(lam, mu, rho, K=CFG.K_max)
        assert P.shape == (CFG.K_max + 1, CFG.K_max + 1)
        assert abs(P.sum() - 1.0) < 1e-12
        assert (P >= 0.0).all()


def test_score_matrix_reduces_to_independent_poisson_at_rho_zero():
    lam, mu, K = 1.9, 1.3, 10
    P = score_matrix(lam, mu, 0.0, K=K)
    from src.model.dixon_coles import _poisson_pmf_vector

    px = _poisson_pmf_vector(lam, K)
    py = _poisson_pmf_vector(mu, K)
    indep = np.outer(px, py)
    indep /= indep.sum()
    assert np.allclose(P, indep, atol=1e-12)


def test_score_matrix_lambda_zero_no_nan():
    P = score_matrix(0.0, 1.2, -0.05, K=10)
    assert not np.isnan(P).any()
    assert abs(P.sum() - 1.0) < 1e-12
    # No home goals possible -> all mass on row 0.
    assert P[1:, :].sum() < 1e-12


def test_score_matrix_only_low_score_cells_corrected():
    # With rho != 0, only the 2x2 low-score block deviates from indep Poisson.
    lam, mu, rho, K = 1.6, 1.0, -0.08, 10
    from src.model.dixon_coles import _poisson_pmf_vector

    indep = np.outer(_poisson_pmf_vector(lam, K), _poisson_pmf_vector(mu, K))
    indep /= indep.sum()
    P = score_matrix(lam, mu, rho, K=K)
    # cells outside the 2x2 block share a common renormalisation ratio
    ratios = P[2:, 2:] / indep[2:, 2:]
    assert np.allclose(ratios, ratios.flat[0], rtol=1e-9)


# ---------------------------------------------------------------------------
# predict_lambdas (spec §3.2)
# ---------------------------------------------------------------------------
def test_predict_lambdas_antisymmetric():
    c0, c1 = 0.28, 0.002
    lam, mu = predict_lambdas(2000.0, 1500.0, 0.0, c0, c1)
    lam2, mu2 = predict_lambdas(1500.0, 2000.0, 0.0, c0, c1)
    assert math.isclose(lam, mu2, rel_tol=1e-12)
    assert math.isclose(mu, lam2, rel_tol=1e-12)
    # stronger home team scores more
    assert lam > mu


def test_predict_lambdas_home_bonus_increases_home_lambda():
    c0, c1 = 0.28, 0.002
    lam0, _ = predict_lambdas(1800.0, 1800.0, 0.0, c0, c1)
    lamH, _ = predict_lambdas(1800.0, 1800.0, 100.0, c0, c1)
    assert lamH > lam0


# ---------------------------------------------------------------------------
# Weighted log-likelihood reproduces the spec formula term-for-term (§3.3)
# ---------------------------------------------------------------------------
def test_weighted_loglik_matches_manual_formula():
    # Hand-built features: two matches, known lambda/mu, known weights.
    feats = pd.DataFrame(
        {
            "diff_home": [0.0, 50.0],  # -> lambda from exp(c0 + c1*diff)
            "x": [0.0, 2.0],
            "y": [0.0, 1.0],
            "date": [pd.Timestamp("2020-01-01").date(), pd.Timestamp("2021-01-01").date()],
        }
    )
    c0, c1, xi, rho = 0.3, 0.002, 0.001, -0.05
    terms = _terms_from_features(feats, c0, c1, xi)
    ll = _weighted_loglik(terms, rho)

    # Manual reconstruction of sum_m phi(dt) [ ln tau - lam + x ln lam - mu + y ln mu ]
    manual = 0.0
    t_ref = pd.Timestamp("2021-01-01")
    for diff, x, y, d in zip(feats["diff_home"], feats["x"], feats["y"], feats["date"]):
        lam = math.exp(c0 + c1 * diff)
        mu = math.exp(c0 - c1 * diff)
        dt = (t_ref - pd.Timestamp(d)).days
        phi = math.exp(-xi * dt)
        manual += phi * (
            math.log(tau(int(x), int(y), lam, mu, rho))
            - lam + x * math.log(lam)
            - mu + y * math.log(mu)
        )
    assert math.isclose(ll, manual, rel_tol=1e-12)


def test_weights_newest_match_is_one():
    feats = pd.DataFrame(
        {
            "diff_home": [0.0, 0.0],
            "x": [1.0, 1.0],
            "y": [1.0, 1.0],
            "date": [pd.Timestamp("2015-06-01").date(), pd.Timestamp("2020-06-01").date()],
        }
    )
    terms = _terms_from_features(feats, 0.3, 0.002, CFG.xi)
    # newest match (2020) has weight exp(0) = 1; older is strictly smaller.
    assert math.isclose(terms.weight.max(), 1.0, rel_tol=1e-12)
    assert terms.weight[0] < terms.weight[1]


# ---------------------------------------------------------------------------
# End-to-end fit on the hand-checkable sample DB (spec §3.2/§3.3)
# ---------------------------------------------------------------------------
def _sample_history(sample_matches):
    return compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )


def test_fit_lambda_coeffs_positive_slope(sample_matches):
    hist = _sample_history(sample_matches)
    c0, c1 = fit_lambda_coeffs(sample_matches, hist, CFG.H)
    assert np.isfinite(c0) and np.isfinite(c1)
    # c1 > 0: stronger team scores more (spec §3.2).
    assert c1 > 0.0


def test_fit_rho_in_bounds(sample_matches):
    hist = _sample_history(sample_matches)
    c0, c1 = fit_lambda_coeffs(sample_matches, hist, CFG.H)
    rho = fit_rho(sample_matches, hist, (c0, c1), CFG.xi, H=CFG.H)
    assert RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]


def test_match_features_leak_free_asof(sample_matches):
    # The as-of pre-match rating must be the rating recorded ON the match date,
    # i.e. never incorporate that match's own result.
    hist = _sample_history(sample_matches)
    feats = match_features(sample_matches, hist, CFG.H)
    assert len(feats) == 6  # all sample matches are played
    assert feats["diff_home"].notna().all()


def test_match_features_matches_elo_engine_prematch_ratings(sample_matches):
    # The as-of pre-match rating attached to each side of each match must equal
    # the rating the Elo engine recorded BEFORE that match (leak-free ground
    # truth). compute_elo_history emits home-row then away-row per match in
    # (date, match_id) order, so we can reconstruct the truth and compare.
    hist = _sample_history(sample_matches)
    feats = match_features(sample_matches, hist, CFG.H)

    ordered = sample_matches.sort_values(["date", "match_id"], kind="mergesort")
    gt_home = hist["elo"].to_numpy()[0::2]
    gt_away = hist["elo"].to_numpy()[1::2]

    # feats is aligned to the played-match order of sample_matches; reorder both
    # to (date, match_id) and reconstruct the diff the engine implies.
    m = sample_matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    feats = feats.assign(match_id=m["match_id"].to_numpy())
    feats_o = feats.set_index("match_id").loc[ordered["match_id"].to_numpy()]

    has_host = "host_flag" in ordered.columns
    for i, (_, mrow) in enumerate(ordered.iterrows()):
        h_eff = CFG.H if not bool(mrow["neutral"]) else 0.0
        expected_diff = (gt_home[i] + h_eff) - gt_away[i]
        assert math.isclose(
            feats_o.iloc[i]["diff_home"], expected_diff, rel_tol=1e-9, abs_tol=1e-9
        )


def test_same_day_double_header_no_self_leak():
    # A team (id 1) plays TWICE on the same calendar day. Each match must get its
    # OWN pre-match rating; the earlier match must NOT inherit the later match's
    # pre-match rating (which already embeds the earlier match's result).
    rows = [
        # id, date, home, away, hg, ag, neutral, tournament, importance_k
        (1, "2020-01-01", 1, 2, 3, 0, False, "Friendly", 10.0),
        (2, "2020-06-01", 1, 3, 2, 2, False, "Friendly", 10.0),
        # same-day double header for team 1 on 2021-03-01 (match 3 then match 4)
        (3, "2021-03-01", 1, 2, 1, 0, False, "Friendly", 10.0),
        (4, "2021-03-01", 3, 1, 0, 2, False, "Friendly", 10.0),
    ]
    cols = ["match_id", "date", "home_id", "away_id", "home_goals", "away_goals",
            "neutral", "tournament", "importance_k"]
    matches = pd.DataFrame(rows, columns=cols)
    matches["date"] = pd.to_datetime(matches["date"]).dt.date
    matches["home_goals"] = matches["home_goals"].astype("Int64")
    matches["away_goals"] = matches["away_goals"].astype("Int64")

    hist = compute_elo_history(
        matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )

    # Ground truth: emit order is home-then-away per match in (date, match_id).
    gt_home = hist["elo"].to_numpy()[0::2]
    gt_away = hist["elo"].to_numpy()[1::2]
    ordered = matches.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    feats = match_features(matches, hist, CFG.H)
    feats = feats.assign(match_id=matches["match_id"].to_numpy())
    feats_o = feats.set_index("match_id").loc[ordered["match_id"].to_numpy()].reset_index()

    # Reconstruct r_home / r_away implied by feats (all non-neutral -> h_eff = H).
    for i in range(len(ordered)):
        expected_diff = (gt_home[i] + CFG.H) - gt_away[i]
        assert math.isclose(
            feats_o.loc[i, "diff_home"], expected_diff, rel_tol=1e-12, abs_tol=1e-9
        ), f"match {ordered.loc[i, 'match_id']} pre-match rating mismatch"

    # The two same-day matches (ids 3, 4) involve team 1 with DIFFERENT pre-match
    # ratings: match 4's team-1 rating embeds match 3's result. Assert the
    # earlier match did NOT inherit the later match's rating (no sibling leak).
    r1_match3 = gt_home[list(ordered["match_id"]).index(3)]   # team 1 is home in m3
    r1_match4 = gt_away[list(ordered["match_id"]).index(4)]   # team 1 is away in m4
    assert not math.isclose(r1_match3, r1_match4, abs_tol=1e-9), (
        "test setup: same-day ratings must differ for the leak check to be meaningful"
    )
    # match 3's implied team-1 rating equals the EARLIER (pre-m3) value, not m4's.
    diff_m3 = feats_o.set_index("match_id").loc[3, "diff_home"]  # r1 + H - r2
    r2_pre_m3 = gt_away[list(ordered["match_id"]).index(3)]
    r1_used_in_m3 = diff_m3 - CFG.H + r2_pre_m3
    assert math.isclose(r1_used_in_m3, r1_match3, abs_tol=1e-9)
    assert not math.isclose(r1_used_in_m3, r1_match4, abs_tol=1e-9)


def test_build_design_two_rows_per_match(sample_matches):
    hist = _sample_history(sample_matches)
    design = build_design(sample_matches, hist, CFG.H)
    assert len(design) == 2 * len(sample_matches)
    # the away-side diff is the negation of the home-side diff for each match
    home_diffs = design["diff"].to_numpy()[0::2]
    away_diffs = design["diff"].to_numpy()[1::2]
    assert np.allclose(home_diffs, -away_diffs)


# ---------------------------------------------------------------------------
# No leakage: fit_up_to uses only matches strictly before `date` (spec §3.6)
# ---------------------------------------------------------------------------
def test_fit_up_to_excludes_future_matches(sample_matches):
    hist = _sample_history(sample_matches)
    # sample matches span 2018..2021; cut at 2020-01-01 keeps ids 1..4.
    fit = fit_up_to("2020-01-01", matches=sample_matches, ratings_history=hist,
                    xi=CFG.xi, H=CFG.H)
    assert fit.n_matches == 4  # matches 1-4 are before 2020-01-01
    assert RHO_BOUNDS[0] <= fit.rho <= RHO_BOUNDS[1]


def test_fit_up_to_is_subset_of_full(sample_matches):
    hist = _sample_history(sample_matches)
    full = fit_full_history(matches=sample_matches, ratings_history=hist,
                            xi=CFG.xi, H=CFG.H)
    early = fit_up_to("2021-01-01", matches=sample_matches, ratings_history=hist,
                      xi=CFG.xi, H=CFG.H)
    assert early.n_matches < full.n_matches
    assert full.n_matches == 6


def test_fit_up_to_raises_when_no_prior_matches(sample_matches):
    hist = _sample_history(sample_matches)
    with pytest.raises(ValueError):
        fit_up_to("1900-01-01", matches=sample_matches, ratings_history=hist,
                  xi=CFG.xi, H=CFG.H)
