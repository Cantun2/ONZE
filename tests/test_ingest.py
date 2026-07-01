"""Tests for the ingest layer (data-engineer): canonical gate, K mapping, odds.

These are the data-engineer's own module tests, plus a check that the shared
sample-DB fixture is usable by downstream agents.
"""

from __future__ import annotations

import csv

import pytest

from src.config import load_config
from src.ingest import canonical
from src.ingest.loaders import map_importance_k
from src.ingest.odds import MissingApiKey, OddsClient


# --- canonical name harmonisation (the hard gate) ---------------------------
def test_classic_aliases_resolve_to_same_team():
    pairs = [
        ("Korea Republic", "South Korea"),
        ("USA", "United States"),
        ("IR Iran", "Iran"),
        ("Czechia", "Czech Republic"),
        ("Cabo Verde", "Cape Verde"),
        ("China PR", "China"),
        ("Chinese Taipei", "Taiwan"),
    ]
    for alias, canon in pairs:
        a, c = canonical.resolve(alias), canonical.resolve(canon)
        assert a is not None, f"{alias!r} did not resolve"
        assert a == c, f"{alias!r} != {canon!r}"


def test_gate_zero_unresolved_over_full_martj42_set():
    cfg = load_config()
    path = cfg.paths.raw_dir / "martj42_results.csv"
    if not path.exists():
        pytest.skip("raw martj42 results not downloaded")
    names = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.add(row["home_team"])
            names.add(row["away_team"])
    assert canonical.unresolved_names(names) == []


def test_unknown_name_is_reported_not_guessed():
    assert canonical.resolve("Definitely Not A Country XYZ") is None
    with pytest.raises(KeyError):
        canonical.resolve_strict("Definitely Not A Country XYZ")


def test_team_ids_are_stable_and_unique():
    rows = canonical.build_team_table()
    ids = [r["team_id"] for r in rows]
    assert len(ids) == len(set(ids))
    # deterministic: France resolves to a fixed id across calls
    assert canonical.resolve("France") == canonical.resolve("France")


def test_accent_folding_collides_variants():
    assert canonical.resolve("Curacao") == canonical.resolve("Curaçao")
    assert canonical.resolve("Reunion") == canonical.resolve("Réunion")


# --- importance_k mapping (spec §3.1) ---------------------------------------
def test_importance_k_mapping():
    k = load_config().k_by_tournament
    assert map_importance_k("FIFA World Cup", k) == k["world_cup_finals"]
    assert map_importance_k("FIFA World Cup qualification", k) == k["qualifier"]
    assert map_importance_k("UEFA Euro qualification", k) == k["qualifier"]
    assert map_importance_k("Friendly", k) == k["friendly"]
    assert map_importance_k("Copa América", k) == k["default"]
    assert map_importance_k(None, k) == k["default"]


# --- odds client behaves without a key --------------------------------------
def test_odds_client_constructs_without_key(monkeypatch):
    monkeypatch.delenv("ONZE_ODDS_API_KEY", raising=False)
    client = OddsClient()  # must NOT raise on construction
    with pytest.raises(MissingApiKey):
        client.fetch_h2h_odds()


def test_odds_rows_shaping():
    events = [
        {
            "home_team": "France",
            "away_team": "Brazil",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "France", "price": 2.1},
                                {"name": "Brazil", "price": 3.4},
                                {"name": "Draw", "price": 3.2},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    rows = OddsClient.to_odds_rows(events, {("France", "Brazil"): 42})
    assert len(rows) == 1
    r = rows[0]
    assert r["fixture_id"] == 42
    assert (r["o_home"], r["o_draw"], r["o_away"]) == (2.1, 3.2, 3.4)


# --- sample DB fixture is usable downstream ---------------------------------
def test_sample_db_fixture_has_known_rows(sample_con):
    n_teams = sample_con.execute("SELECT count(*) FROM teams").fetchone()[0]
    n_matches = sample_con.execute("SELECT count(*) FROM matches").fetchone()[0]
    assert n_teams == 4
    assert n_matches == 6
    # a known value: match 1 France 2-1 Brazil
    row = sample_con.execute(
        "SELECT home_goals, away_goals FROM matches WHERE match_id = 1"
    ).fetchone()
    assert row == (2, 1)


def test_sample_db_has_full_schema(sample_con):
    tables = {
        r[0] for r in sample_con.execute("SHOW TABLES").fetchall()
    }
    assert {
        "teams", "matches", "ratings_history", "fixtures", "predictions", "odds",
    } <= tables
