"""Tests for the Elo rating engine (src/ratings/elo.py, spec §3.1 + §3.6).

These lock in the two properties that matter most for ONZE:

* **No leakage** — the rating stored for a match is a function of strictly
  earlier matches only (spec §3.6). This is asserted directly by recomputing
  the ratings independently and checking the stored value equals the pre-match
  value for every match.
* **Correct maths** — the §3.1 update rules on the hand-checkable sample DB.
"""

from __future__ import annotations

import math

import pandas as pd

from src.config import load_config
from src.ratings.elo import (
    compute_elo_history,
    elo_1x2,
    expected_home_score,
    goal_diff_multiplier,
    latest_ratings,
    rating_as_of,
)

CFG = load_config()


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(-1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == (11 + 3) / 8
    assert goal_diff_multiplier(-5) == (11 + 5) / 8


def test_expected_home_score_symmetry_and_bonus():
    # Even ratings, no bonus -> 0.5.
    assert expected_home_score(1500, 1500, 0.0) == 0.5
    # Home bonus makes We > 0.5.
    assert expected_home_score(1500, 1500, 100.0) > 0.5
    # 400-point edge without bonus -> ~0.909.
    assert math.isclose(expected_home_score(1900, 1500, 0.0), 10 / 11, rel_tol=1e-9)


def test_no_leakage_stored_is_pre_match(sample_matches):
    """The stored rating for every match must equal its independently
    recomputed PRE-match rating (spec §3.6). If the engine leaked a match's own
    result into its stored rating, this would fail."""
    h = compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )
    stored = {(int(r.team_id), r.date): r.elo for r in h.itertuples()}

    ratings: dict[int, float] = {}
    base = CFG.base_rating
    for r in sample_matches.sort_values(["date", "match_id"]).itertuples(index=False):
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue
        rh = ratings.get(int(r.home_id), base)
        ra = ratings.get(int(r.away_id), base)
        # stored == pre-match, for both teams
        assert stored[(int(r.home_id), r.date)] == rh
        assert stored[(int(r.away_id), r.date)] == ra
        heff = 0.0 if r.neutral else CFG.H
        we = expected_home_score(rh, ra, heff)
        w = 1.0 if r.home_goals > r.away_goals else (0.0 if r.home_goals < r.away_goals else 0.5)
        g = goal_diff_multiplier(int(r.home_goals) - int(r.away_goals))
        ch = float(r.importance_k) * g * (w - we)
        ratings[int(r.home_id)] = rh + ch
        ratings[int(r.away_id)] = ra - ch


def test_zero_sum_conservation(sample_matches):
    """Every update is zero-sum, so the total rating mass is conserved
    (n_teams * base_rating)."""
    h = compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )
    # Reconstruct final ratings to check conservation.
    ratings: dict[int, float] = {}
    base = CFG.base_rating
    teams = set()
    for r in sample_matches.sort_values(["date", "match_id"]).itertuples(index=False):
        teams.update([int(r.home_id), int(r.away_id)])
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue
        rh = ratings.get(int(r.home_id), base)
        ra = ratings.get(int(r.away_id), base)
        heff = 0.0 if r.neutral else CFG.H
        we = expected_home_score(rh, ra, heff)
        w = 1.0 if r.home_goals > r.away_goals else (0.0 if r.home_goals < r.away_goals else 0.5)
        g = goal_diff_multiplier(int(r.home_goals) - int(r.away_goals))
        ch = float(r.importance_k) * g * (w - we)
        ratings[int(r.home_id)] = rh + ch
        ratings[int(r.away_id)] = ra - ch
    assert math.isclose(sum(ratings.values()), len(teams) * base, abs_tol=1e-6)


def test_first_match_hand_computation(sample_matches):
    """Match 1 (France 2-1 Brazil, non-neutral WC, K=60): both start at 1500,
    We = 1/(1+10^(-100/400)), G=1, W=1."""
    h = compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )
    we = 1.0 / (1.0 + 10 ** (-100 / 400))
    change = 60 * 1.0 * (1.0 - we)
    m1 = h[h["date"] == pd.to_datetime("2018-06-16").date()]
    # both stored pre-match ratings are the base
    assert set(m1["elo"]) == {1500.0}
    # France's rating entering match 3 should be its post-match-1 value
    m3 = h[(h["date"] == pd.to_datetime("2019-03-22").date()) & (h["team_id"] == 1)]
    assert math.isclose(m3["elo"].iloc[0], 1500.0 + change, rel_tol=1e-9)


def test_null_goals_skipped():
    cols = ["match_id", "date", "home_id", "away_id", "home_goals",
            "away_goals", "neutral", "tournament", "importance_k"]
    rows = [
        (1, "2020-01-01", 1, 2, 2, 0, False, "Friendly", 10.0),
        (2, "2020-02-01", 1, 2, None, None, False, "Friendly", 10.0),  # unplayed
        (3, "2020-03-01", 1, 2, 1, 1, False, "Friendly", 10.0),
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    h = compute_elo_history(df, H=CFG.H, k_by_tournament=CFG.k_by_tournament)
    # 2 played matches -> 4 rows; the null row contributes none.
    assert len(h) == 4
    assert pd.to_datetime("2020-02-01").date() not in set(h["date"])


def test_elo_1x2_valid_distribution():
    for args in [(2000, 1500, 100), (1500, 1500, 0), (1400, 1900, 0), (1500, 1500, 100)]:
        p_home, p_draw, p_away = elo_1x2(*args)
        assert all(p >= 0 for p in (p_home, p_draw, p_away))
        assert math.isclose(p_home + p_draw + p_away, 1.0, abs_tol=1e-9)
    # Symmetric fixture: home and away equal.
    ph, pd_, pa = elo_1x2(1500, 1500, 0)
    assert math.isclose(ph, pa, abs_tol=1e-12)
    # Stronger home -> higher home prob.
    assert elo_1x2(2000, 1500, 100)[0] > elo_1x2(1500, 2000, 0)[0]


def test_rating_as_of_leak_free(sample_matches):
    h = compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )
    # Strictly before the first recorded match -> base rating.
    assert rating_as_of(1, "2018-06-16", history=h, base_rating=CFG.base_rating) == CFG.base_rating
    # Never returns a rating that used a match on/after the query date.
    val = rating_as_of(1, "2019-01-01", history=h, base_rating=CFG.base_rating)
    assert val == CFG.base_rating  # France's only pre-2019 entry is its pre-match-1 (base)
    # Unknown team -> base.
    assert rating_as_of(999, "2025-01-01", history=h, base_rating=CFG.base_rating) == CFG.base_rating


def test_latest_ratings(sample_matches):
    h = compute_elo_history(
        sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament,
        base_rating=CFG.base_rating,
    )
    last = latest_ratings(h)
    assert set(last.columns) == {"team_id", "elo"}
    assert len(last) == last["team_id"].nunique()


def test_deterministic(sample_matches):
    a = compute_elo_history(sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament)
    b = compute_elo_history(sample_matches, H=CFG.H, k_by_tournament=CFG.k_by_tournament)
    pd.testing.assert_frame_equal(a, b)
