"""Tests for the predict package (src/predict/, spec §3.4, §3.5, §1/§5).

Covers the prediction-engineer definition of done:

* Derived 1N2 from the matrix equals the direct triangle/diagonal sums.
* over/under and BTTS validated against **hand-computed** small cases.
* The single-fixture pipeline runs end-to-end and serialises to ``predictions``
  (matrix_json round-trips losslessly).
* The bracket Monte-Carlo is seeded/reproducible and its per-round and champion
  probabilities sum to the right totals (champion probs over all teams == 1).

The markets tests build the matrix by hand so the expected numbers are checkable
by eye; the pipeline/bracket tests use a small self-contained DuckDB store with a
full QF->SF->Final bracket and known coefficients.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.model.dixon_coles import score_matrix
from src.model.knockout import resolve_stage
from src.predict.markets import btts, derive_markets, one_x_two, over_under
from src.predict import bracket as bracket_mod
from src.predict import fixture as fixture_mod
from src.ingest.schema import _DDL


# ---------------------------------------------------------------------------
# markets.py — hand-computed small cases (definition of done)
# ---------------------------------------------------------------------------
def _tiny_matrix() -> np.ndarray:
    """A hand-built 3x3 pmf with easy-to-sum cells (rows=home x, cols=away y).

        y=0    y=1    y=2
    x=0 0.10   0.10   0.05
    x=1 0.15   0.30   0.05
    x=2 0.10   0.10   0.05

    Sums to 1.00. Chosen so every derived market is trivially checkable by hand.
    """
    P = np.array(
        [
            [0.10, 0.10, 0.05],
            [0.15, 0.30, 0.05],
            [0.10, 0.10, 0.05],
        ]
    )
    assert abs(P.sum() - 1.0) < 1e-12
    return P


def test_1x2_equals_direct_sums_hand():
    P = _tiny_matrix()
    p_home, p_draw, p_away = one_x_two(P)
    # home (x>y): (1,0)=.15 (2,0)=.10 (2,1)=.10 -> .35
    # draw (x=y): (0,0)=.10 (1,1)=.30 (2,2)=.05 -> .45
    # away (x<y): (0,1)=.10 (0,2)=.05 (1,2)=.05 -> .20
    assert p_home == pytest.approx(0.35)
    assert p_draw == pytest.approx(0.45)
    assert p_away == pytest.approx(0.20)
    assert p_home + p_draw + p_away == pytest.approx(1.0)


def test_1x2_matches_knockout_resolve_stage():
    """Derived 1N2 must equal knockout-engineer's resolve_stage (same partition)."""
    P = score_matrix(1.9, 1.1, -0.05, K=10)
    p_home, p_draw, p_away = one_x_two(P)
    kh, ka, kd = resolve_stage(P)  # (home, away, level)
    assert p_home == pytest.approx(kh)
    assert p_away == pytest.approx(ka)
    assert p_draw == pytest.approx(kd)


def test_1x2_equals_direct_sums_real_matrix():
    """On a real score matrix the derived 1N2 equals the direct triangle sums."""
    P = score_matrix(2.1, 0.8, -0.04, K=10)
    p_home, p_draw, p_away = one_x_two(P)
    assert p_home == pytest.approx(float(np.tril(P, -1).sum()))
    assert p_draw == pytest.approx(float(np.trace(P)))
    assert p_away == pytest.approx(float(np.triu(P, 1).sum()))


def test_over_under_hand_computed():
    P = _tiny_matrix()
    ou = over_under(P)
    # totals by x+y:
    #  t=0: (0,0)=.10
    #  t=1: (0,1)+(1,0)=.10+.15=.25
    #  t=2: (0,2)+(1,1)+(2,0)=.05+.30+.10=.45
    #  t=3: (1,2)+(2,1)=.05+.10=.15
    #  t=4: (2,2)=.05
    # over 0.5 = 1 - t0 = .90
    assert ou["0.5"]["over"] == pytest.approx(0.90)
    # over 1.5 = t2+t3+t4 = .45+.15+.05 = .65
    assert ou["1.5"]["over"] == pytest.approx(0.65)
    # over 2.5 = t3+t4 = .20
    assert ou["2.5"]["over"] == pytest.approx(0.20)
    # over 3.5 = t4 = .05
    assert ou["3.5"]["over"] == pytest.approx(0.05)
    # over 4.5 = 0 (no total > 4)
    assert ou["4.5"]["over"] == pytest.approx(0.0)
    for line in ou:
        assert ou[line]["over"] + ou[line]["under"] == pytest.approx(1.0)


def test_btts_hand_computed():
    P = _tiny_matrix()
    # P(X=0) = row0 sum = .10+.10+.05 = .25
    # P(Y=0) = col0 sum = .10+.15+.10 = .35
    # P(0,0) = .10
    # btts = 1 - .25 - .35 + .10 = .50
    assert btts(P) == pytest.approx(0.50)


def test_btts_formula_matches_direct_count():
    """BTTS from the §3.4 formula equals a direct sum over x>=1 and y>=1 cells."""
    P = score_matrix(1.7, 1.3, -0.05, K=10)
    direct = float(P[1:, 1:].sum())  # both score at least one
    assert btts(P) == pytest.approx(direct)


def test_over_under_matches_direct_count():
    P = score_matrix(2.4, 1.6, -0.03, K=10)
    ou = over_under(P)
    K = P.shape[0] - 1
    xs, ys = np.indices(P.shape)
    totals = xs + ys
    for line_str, vals in ou.items():
        line = float(line_str)
        direct_over = float(P[totals > line].sum())
        assert vals["over"] == pytest.approx(direct_over)


def test_derive_markets_full_shape():
    P = score_matrix(1.8, 1.1, -0.05, K=10)
    m = derive_markets(P)
    assert set(m) >= {
        "p_home", "p_draw", "p_away", "most_likely_score",
        "most_likely_score_str", "top5_scores", "over_under", "btts",
    }
    # top-5 are sorted by descending probability and are actual cells of P.
    probs = [p for (_, p) in m["top5_scores"]]
    assert probs == sorted(probs, reverse=True)
    assert len(m["top5_scores"]) == 5
    (x, y), top_p = m["top5_scores"][0]
    assert (x, y) == m["most_likely_score"]
    assert top_p == pytest.approx(float(P.max()))
    # every market probability is in [0, 1]
    assert 0.0 <= m["btts"] <= 1.0
    assert m["p_home"] + m["p_draw"] + m["p_away"] == pytest.approx(1.0)


def test_derive_markets_rejects_bad_matrix():
    with pytest.raises(ValueError):
        derive_markets(np.array([[1.0, 2.0, 3.0]]))  # not square
    with pytest.raises(ValueError):
        derive_markets(np.array([[0.5, -0.1], [0.3, 0.3]]))  # negative


# ---------------------------------------------------------------------------
# fixture.py + bracket.py — a self-contained bracket DB
# ---------------------------------------------------------------------------
@pytest.fixture()
def bracket_db(tmp_path):
    """A small DuckDB store with 8 teams, coeffs, and a full QF->SF->Final bracket.

    Team strengths are spread so the bracket has a clear favourite, letting us
    assert monotone reach probabilities.
    """
    db = tmp_path / "bracket.duckdb"
    con = duckdb.connect(str(db))
    con.execute(_DDL)

    teams = pd.DataFrame(
        [
            {"team_id": 310, "name_canonical": "United States", "confederation": "CONCACAF", "elo_current": 1877.0},
            {"team_id": 39, "name_canonical": "Brazil", "confederation": "CONMEBOL", "elo_current": 2034.0},
            {"team_id": 102, "name_canonical": "France", "confederation": "UEFA", "elo_current": 2131.0},
            {"team_id": 13, "name_canonical": "Argentina", "confederation": "CONMEBOL", "elo_current": 2131.0},
            {"team_id": 91, "name_canonical": "England", "confederation": "UEFA", "elo_current": 2026.0},
            {"team_id": 277, "name_canonical": "Spain", "confederation": "UEFA", "elo_current": 2071.0},
            {"team_id": 198, "name_canonical": "Netherlands", "confederation": "UEFA", "elo_current": 2009.0},
            {"team_id": 227, "name_canonical": "Portugal", "confederation": "UEFA", "elo_current": 1989.0},
        ]
    )
    con.register("_t", teams)
    con.execute("INSERT INTO teams SELECT * FROM _t")
    con.unregister("_t")

    # Fitted-ish coefficients (mirroring the real dc-v1 fit).
    con.execute(
        """CREATE TABLE IF NOT EXISTS model_coeffs (
            model_version VARCHAR, c0 DOUBLE, c1 DOUBLE, rho DOUBLE, xi DOUBLE,
            loglik DOUBLE, n_matches BIGINT, fit_up_to DATE, created_at TIMESTAMP)"""
    )
    con.execute(
        "INSERT INTO model_coeffs VALUES ('dc-v1', 0.2845, 0.001991, -0.0525, "
        "0.00095, -4408.0, 49484, NULL, now())"
    )

    fixtures = [
        (0, _dt.date(2026, 7, 11), 310, 39, "Quarter-final", False, True),
        (1, _dt.date(2026, 7, 11), 102, 13, "Quarter-final", True, False),
        (2, _dt.date(2026, 7, 12), 91, 277, "Quarter-final", True, False),
        (3, _dt.date(2026, 7, 12), 198, 227, "Quarter-final", True, False),
        (4, _dt.date(2026, 7, 14), 39, 102, "Semi-final", True, False),
        (5, _dt.date(2026, 7, 15), 277, 198, "Semi-final", True, False),
        (6, _dt.date(2026, 7, 18), 102, 277, "Third place", True, False),
        (7, _dt.date(2026, 7, 19), 39, 198, "Final", True, False),
    ]
    for f in fixtures:
        con.execute("INSERT INTO fixtures VALUES (?, ?, ?, ?, ?, ?, ?)", list(f))
    con.commit()
    con.close()
    return db


def test_predict_fixture_pipeline(bracket_db):
    con = duckdb.connect(str(bracket_db))
    try:
        r = fixture_mod.predict_fixture(0, con=con)  # USA vs Brazil, host USA
    finally:
        con.close()
    # 1N2 is a valid distribution summing to 1.
    assert r["p_home"] + r["p_draw"] + r["p_away"] == pytest.approx(1.0)
    # Matrix is a proper pmf of the configured size.
    P = np.asarray(r["matrix"])
    assert P.shape == (11, 11)
    assert P.sum() == pytest.approx(1.0)
    # Knockout stage -> advance probs attached and summing to 1.
    assert "advance" in r
    adv = r["advance"]
    assert adv["p_home_advance"] + adv["p_away_advance"] == pytest.approx(1.0)
    # Brazil (away, stronger) should be favoured to advance despite USA hosting.
    assert adv["p_away_advance"] > adv["p_home_advance"]
    # Host bonus is applied (non-neutral effective bonus for USA host_flag).
    assert r["h_eff"] == 100.0


def test_matrix_json_round_trips(bracket_db):
    con = duckdb.connect(str(bracket_db))
    try:
        r = fixture_mod.predict_fixture(1, con=con)
        row = fixture_mod._row_for_predictions(r)
    finally:
        con.close()
    # matrix_json is the 3rd column; decode and compare to the original matrix.
    matrix_json = row[2]
    P_back = fixture_mod.matrix_from_json(matrix_json)
    assert np.allclose(P_back, np.asarray(r["matrix"]), atol=1e-15)
    # most_likely_score column agrees with the markets tuple.
    x, y = r["most_likely_score"]
    assert row[6] == f"{x}-{y}"


def test_write_predictions_persists_and_reads_back(bracket_db):
    con = duckdb.connect(str(bracket_db))
    try:
        results = [fixture_mod.predict_fixture(fid, con=con) for fid in range(8)]
        n = fixture_mod.write_predictions(con, results)
        con.commit()
        assert n == 8
        got = con.execute(
            "SELECT fixture_id, matrix_json, p_home, most_likely_score "
            "FROM predictions ORDER BY fixture_id"
        ).fetchdf()
        assert len(got) == 8
        # matrix_json round-trips from the DB for fixture 0.
        row0 = got[got["fixture_id"] == 0].iloc[0]
        P = fixture_mod.matrix_from_json(row0["matrix_json"])
        assert P.sum() == pytest.approx(1.0)
        # Idempotent: re-writing does not duplicate rows.
        fixture_mod.write_predictions(con, results)
        con.commit()
        assert con.execute("SELECT count(*) FROM predictions").fetchone()[0] == 8
    finally:
        con.close()


def test_is_knockout_stage():
    assert fixture_mod.is_knockout_stage("Quarter-final")
    assert fixture_mod.is_knockout_stage("Final")
    assert fixture_mod.is_knockout_stage("semi-final")
    assert not fixture_mod.is_knockout_stage("Group stage")
    assert not fixture_mod.is_knockout_stage(None)


# ---------------------------------------------------------------------------
# bracket.py — Monte-Carlo simulation invariants
# ---------------------------------------------------------------------------
def test_bracket_sim_sums_and_monotone(bracket_db):
    con = duckdb.connect(str(bracket_db))
    try:
        df = bracket_mod.simulate_tournament(n_sims=40_000, seed=123, con=con)
    finally:
        con.close()

    # 8 quarter-finalists.
    assert len(df) == 8
    # Champion probabilities over all teams sum to 1 (one champion per sim).
    assert df["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)
    # Exactly two finalists and four semi-finalists per sim.
    assert df["p_reach_final"].sum() == pytest.approx(2.0, abs=1e-9)
    assert df["p_reach_semi"].sum() == pytest.approx(4.0, abs=1e-9)
    # Monotone per team: champion <= reach_final <= reach_semi.
    assert (df["p_champion"] <= df["p_reach_final"] + 1e-12).all()
    assert (df["p_reach_final"] <= df["p_reach_semi"] + 1e-12).all()
    # Every semi reach prob is a valid probability.
    assert ((df["p_reach_semi"] >= 0) & (df["p_reach_semi"] <= 1)).all()


def test_bracket_sim_reproducible(bracket_db):
    con = duckdb.connect(str(bracket_db))
    try:
        a = bracket_mod.simulate_tournament(n_sims=20_000, seed=999, con=con)
        b = bracket_mod.simulate_tournament(n_sims=20_000, seed=999, con=con)
        c = bracket_mod.simulate_tournament(n_sims=20_000, seed=1000, con=con)
    finally:
        con.close()
    # Same seed -> bit-for-bit identical champion probabilities.
    a_sorted = a.sort_values("team_id").reset_index(drop=True)
    b_sorted = b.sort_values("team_id").reset_index(drop=True)
    assert np.array_equal(a_sorted["p_champion"].to_numpy(), b_sorted["p_champion"].to_numpy())
    # A different seed generally gives at least slightly different estimates.
    c_sorted = c.sort_values("team_id").reset_index(drop=True)
    assert not np.array_equal(a_sorted["p_champion"].to_numpy(), c_sorted["p_champion"].to_numpy())


# ---------------------------------------------------------------------------
# effective_elo() — the live news-adjustment overlay (prediction-time only)
# ---------------------------------------------------------------------------
def test_effective_elo_applies_delta_and_defaults_zero(bracket_db):
    """`effective_elo` adds the configured delta and defaults to 0 when unset."""
    import dataclasses

    con = duckdb.connect(str(bracket_db))
    try:
        cfg = fixture_mod.load_config()
        overlay_cfg = dataclasses.replace(
            cfg,
            raw={
                **cfg.raw,
                "news_adjustments": {
                    "Brazil": {"delta": -35, "note": "key players out"},
                },
            },
        )
        base_brazil = fixture_mod._team_elo(con, 39)  # Brazil (bracket_db teams)
        assert fixture_mod.effective_elo(con, 39, overlay_cfg) == pytest.approx(
            base_brazil - 35
        )
        delta, note = fixture_mod.news_delta(39, overlay_cfg)
        assert delta == pytest.approx(-35)
        assert note == "key players out"

        # A team with no entry in news_adjustments defaults to a 0 delta.
        base_france = fixture_mod._team_elo(con, 102)  # France
        assert fixture_mod.effective_elo(con, 102, overlay_cfg) == pytest.approx(
            base_france
        )
        assert fixture_mod.news_delta(102, overlay_cfg) == (0.0, None)
    finally:
        con.close()


def test_predict_fixture_surfaces_news_adjustment(bracket_db):
    """`predict_fixture`'s lambdas use the overlay and it's surfaced honestly."""
    import dataclasses

    con = duckdb.connect(str(bracket_db))
    try:
        cfg = fixture_mod.load_config()
        overlay_cfg = dataclasses.replace(
            cfg,
            raw={
                **cfg.raw,
                "news_adjustments": {"Brazil": {"delta": -35, "note": "injuries"}},
            },
        )
        r = fixture_mod.predict_fixture(0, con=con, cfg=overlay_cfg)  # USA vs Brazil
        base_brazil = fixture_mod._team_elo(con, 39)
        # The lambda-feeding elo_away is the *effective* (overlay-adjusted) elo.
        assert r["elo_away"] == pytest.approx(base_brazil - 35)
        assert r["news_adjustment"]["away"] == {"delta": -35, "note": "injuries"}
        # Home side (USA) has no news entry -> zero delta, no note.
        assert r["news_adjustment"]["home"] == {"delta": 0.0, "note": None}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# played_result() — deterministic resolution of already-played KO ties
# ---------------------------------------------------------------------------
def test_played_result_resolves_decisive_score(bracket_db):
    """A completed match with a decisive score resolves to the higher scorer."""
    con = duckdb.connect(str(bracket_db))
    try:
        con.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [900001, _dt.date(2026, 7, 11), 310, 39, 0, 2, True, "FIFA World Cup", 60.0],
        )
        winner = bracket_mod.played_result(con, 310, 39, _dt.date(2026, 7, 11))
        assert winner == 39  # Brazil won 2-0
        # Order-agnostic: querying with the pair flipped gives the same winner.
        winner_flipped = bracket_mod.played_result(con, 39, 310, _dt.date(2026, 7, 11))
        assert winner_flipped == 39
    finally:
        con.close()


def test_played_result_resolves_shootout_from_csv(bracket_db):
    """A level score resolves via the martj42 shootout record (documented data)."""
    con = duckdb.connect(str(bracket_db))
    try:
        # Real 2026 shootout on record: Netherlands 1-1 Morocco, Morocco won on pens.
        netherlands_id = 198  # already in bracket_db
        morocco_id = 193
        con.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [900002, _dt.date(2026, 6, 29), netherlands_id, morocco_id, 1, 1, True,
             "FIFA World Cup", 60.0],
        )
        winner = bracket_mod.played_result(
            con, netherlands_id, morocco_id, _dt.date(2026, 6, 29)
        )
        assert winner == morocco_id
    finally:
        con.close()


def test_played_result_none_when_unplayed_or_no_matches_table(bracket_db):
    """Unplayed ties, and self-contained DBs without matches rows, degrade to None."""
    con = duckdb.connect(str(bracket_db))
    try:
        # bracket_db's matches table exists but is empty -> nothing to resolve.
        assert bracket_mod.played_result(con, 310, 39, _dt.date(2026, 7, 11)) is None
    finally:
        con.close()


def test_bracket_sim_uses_played_result_deterministically(bracket_db):
    """A base tie with a recorded result always sends the actual winner through."""
    con = duckdb.connect(str(bracket_db))
    try:
        # QF0 (fixture_id=0) is USA(310) vs Brazil(39) on 2026-07-11; force Brazil in.
        con.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [900003, _dt.date(2026, 7, 11), 310, 39, 0, 3, True, "FIFA World Cup", 60.0],
        )
        df = bracket_mod.simulate_tournament(n_sims=2_000, seed=42, con=con)
        usa = df[df["team_id"] == 310].iloc[0]
        brazil = df[df["team_id"] == 39].iloc[0]
        # Brazil reaches the semi-final in every single simulation (deterministic QF).
        assert brazil["p_reach_semi"] == pytest.approx(1.0)
        # USA is eliminated in the QF in every simulation.
        assert usa["p_reach_semi"] == pytest.approx(0.0)
    finally:
        con.close()


def test_bracket_favourite_leads(bracket_db):
    """The strongest teams (France/Argentina, 2131) should top champion odds."""
    con = duckdb.connect(str(bracket_db))
    try:
        df = bracket_mod.simulate_tournament(n_sims=40_000, seed=7, con=con)
    finally:
        con.close()
    top = df.sort_values("p_champion", ascending=False).iloc[0]
    # The favourite is one of the 2131-Elo sides on the QF0/QF1 half.
    assert top["team"] in {"France", "Argentina"}
    # The weakest side (USA, 1877, and facing Brazil) has the lowest champ odds.
    usa = df[df["team"] == "United States"].iloc[0]
    assert usa["p_champion"] == df["p_champion"].min()
