"""Smoke tests for the ONZE serving API (spec §4.4).

Hits every one of the five §4.4 endpoints against the *real, already-populated*
DuckDB store (8 seeded fixtures, 8 predictions) via FastAPI's ``TestClient``,
asserting status codes and response shapes. Endpoints that depend on artifacts
which may not exist yet (the eval backtest report) are guarded so the suite is
robust: they must return either a clean ``200`` or a clean ``503`` — never a
crash.

Also verifies the load-bearing invariant that the P(x,y) matrix round-trips
through JSON without precision loss (spec DoD).
"""

from __future__ import annotations

import json

import duckdb
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import load_config

client = TestClient(app)


def _db_has_predictions() -> bool:
    cfg = load_config()
    if not cfg.paths.db_path.exists():
        return False
    con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
    try:
        return con.execute("SELECT count(*) FROM predictions").fetchone()[0] > 0
    finally:
        con.close()


requires_db = pytest.mark.skipif(
    not _db_has_predictions(),
    reason="real DB with predictions not populated; run `python -m src.predict.fixture`",
)


def test_root_index():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "ONZE API"
    assert "GET /fixtures" in body["endpoints"]


@requires_db
def test_fixtures_endpoint():
    r = client.get("/fixtures")
    assert r.status_code == 200
    fixtures = r.json()
    assert isinstance(fixtures, list) and len(fixtures) > 0
    fx = fixtures[0]
    for key in ("fixture_id", "home_team", "away_team", "stage", "neutral", "knockout"):
        assert key in fx
    # team names must be joined (not bare ids)
    assert isinstance(fx["home_team"], str)


@requires_db
def test_predict_endpoint_and_matrix_roundtrip():
    # pick a real fixture id from /fixtures
    fixtures = client.get("/fixtures").json()
    fid = fixtures[0]["fixture_id"]

    r = client.get(f"/predict/{fid}")
    assert r.status_code == 200
    body = r.json()

    # shape
    for key in ("matrix", "p_home", "p_draw", "p_away", "most_likely_score",
                "top5_scores", "over_under", "btts"):
        assert key in body

    matrix = body["matrix"]
    assert isinstance(matrix, list) and len(matrix) == len(matrix[0])  # square

    # 1N2 partitions the matrix -> sums to ~1
    assert abs(body["p_home"] + body["p_draw"] + body["p_away"] - 1.0) < 1e-9

    # matrix mass ~1 (renormalised P(x,y))
    total = sum(sum(row) for row in matrix)
    assert abs(total - 1.0) < 1e-9

    # ---- matrix round-trips JSON without precision loss (spec DoD) ----
    cfg = load_config()
    con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
    try:
        stored_json = con.execute(
            "SELECT matrix_json FROM predictions WHERE fixture_id = ? LIMIT 1", [fid]
        ).fetchone()[0]
    finally:
        con.close()
    stored = json.loads(stored_json)
    # the API-served matrix must be bit-identical to the stored one
    assert stored == matrix


@requires_db
def test_predict_knockout_has_advance():
    fixtures = client.get("/fixtures").json()
    ko = next((f for f in fixtures if f["knockout"]), None)
    if ko is None:
        pytest.skip("no knockout fixture in seeded set")
    body = client.get(f"/predict/{ko['fixture_id']}").json()
    assert body["advance"] is not None
    adv = body["advance"]
    assert abs(adv["p_home_advance"] + adv["p_away_advance"] - 1.0) < 1e-6


def test_predict_unknown_fixture_404():
    r = client.get("/predict/999999")
    assert r.status_code == 404


@requires_db
def test_bracket_endpoint():
    r = client.get("/bracket")
    assert r.status_code == 200
    body = r.json()
    assert body["n_sims"] > 0
    teams = body["teams"]
    assert isinstance(teams, list) and len(teams) > 0
    for t in teams:
        for key in ("team_id", "team", "p_reach_semi", "p_reach_final", "p_champion"):
            assert key in t
    # champion probabilities over all teams ~ 1 (one champion per sim)
    champ_sum = sum(t["p_champion"] for t in teams)
    assert abs(champ_sum - 1.0) < 0.02  # Monte-Carlo tolerance
    # monotone: champion <= reach_final <= reach_semi per team
    for t in teams:
        assert t["p_champion"] <= t["p_reach_final"] + 1e-9
        assert t["p_reach_final"] <= t["p_reach_semi"] + 1e-9


def test_backtest_endpoint_200_or_503():
    """Eval report may not exist yet: accept a clean 200 or a clean 503."""
    r = client.get("/eval/backtest?from=2010")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert isinstance(body["results"], list) and len(body["results"]) > 0
        assert "source" in body
        # optional calibration/ece pass-throughs must be present as keys (may be null)
        for key in ("calibration", "ece", "xi_chosen", "xi_half_life_days",
                    "caveat", "generated_from", "n_matches"):
            assert key in body
    else:
        # must be a clean, informative 503 (not a crash / 500)
        assert "detail" in r.json()


def test_backtest_calibration_passthrough():
    """When the artifact carries a calibration curve + ece, pass them through."""
    r = client.get("/eval/backtest?from=2020")
    if r.status_code != 200:
        pytest.skip("no eval report artifact available")
    body = r.json()
    cal = body["calibration"]
    assert isinstance(cal, list) and len(cal) > 0
    b0 = cal[0]
    for key in ("bin_lo", "bin_hi", "p_pred", "p_obs", "n"):
        assert key in b0
    assert isinstance(body["ece"], float)


@requires_db
def test_refresh_endpoint():
    r = client.post("/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["predictions_written"] > 0
    assert body["caches_cleared"] is True
