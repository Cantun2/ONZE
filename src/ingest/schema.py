"""DuckDB schema (spec §2.2) — creation + population helpers.

The schema mirrors ONZE_spec.md §2.2 field-for-field:

    teams(team_id, name_canonical, confederation, elo_current)
    matches(match_id, date, home_id, away_id, home_goals, away_goals,
            neutral, tournament, importance_k)
    ratings_history(team_id, date, elo)            # owned by ratings/
    fixtures(fixture_id, date, home_id, away_id, stage, neutral, host_flag)
    predictions(fixture_id, model_version, matrix_json, p_home, p_draw,
                p_away, most_likely_score, created_at)   # owned by predict/
    odds(fixture_id, bookmaker, o_home, o_draw, o_away, captured_at)  # by ingest/odds

The ingest owns creating the tables and populating ``teams`` + ``matches``.
``ratings_history`` / ``predictions`` / ``odds`` are created empty and filled by
their owning modules. ``fixtures`` gets a small ILLUSTRATIVE seed so downstream
KO / bracket / API code has something to run against before the real 2026 draw
is wired.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.config import Config
from src.ingest import canonical

_DDL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id        INTEGER PRIMARY KEY,
    name_canonical VARCHAR NOT NULL,
    confederation  VARCHAR,
    elo_current    DOUBLE
);

CREATE TABLE IF NOT EXISTS matches (
    match_id     BIGINT PRIMARY KEY,
    date         DATE,
    home_id      INTEGER,
    away_id      INTEGER,
    home_goals   INTEGER,     -- NULL when unknown (never fabricated)
    away_goals   INTEGER,
    neutral      BOOLEAN,
    tournament   VARCHAR,
    importance_k DOUBLE
);

CREATE TABLE IF NOT EXISTS ratings_history (
    team_id INTEGER,
    date    DATE,
    elo     DOUBLE
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id INTEGER PRIMARY KEY,
    date       DATE,
    home_id    INTEGER,
    away_id    INTEGER,
    stage      VARCHAR,
    neutral    BOOLEAN,
    host_flag  BOOLEAN
);

CREATE TABLE IF NOT EXISTS predictions (
    fixture_id        INTEGER,
    model_version     VARCHAR,
    matrix_json       VARCHAR,
    p_home            DOUBLE,
    p_draw            DOUBLE,
    p_away            DOUBLE,
    most_likely_score VARCHAR,
    created_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds (
    fixture_id  INTEGER,
    bookmaker   VARCHAR,
    o_home      DOUBLE,
    o_draw      DOUBLE,
    o_away      DOUBLE,
    captured_at TIMESTAMP
);
"""


def create_schema(cfg: Config) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB store at ``cfg.paths.db_path`` and ensure the schema."""
    db_path = cfg.paths.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_DDL)
    return con


def populate_teams(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    """Replace the ``teams`` table with the canonical registry rows."""
    df = pd.DataFrame(rows, columns=["team_id", "name_canonical", "confederation", "elo_current"])
    con.execute("DELETE FROM teams")
    con.register("_teams_df", df)
    con.execute("INSERT INTO teams SELECT * FROM _teams_df")
    con.unregister("_teams_df")
    return len(df)


def populate_matches(con: duckdb.DuckDBPyConnection, matches: pd.DataFrame) -> int:
    """Insert cleaned matches (only the §2.2 columns) into ``matches``."""
    cols = [
        "match_id", "date", "home_id", "away_id", "home_goals", "away_goals",
        "neutral", "tournament", "importance_k",
    ]
    df = matches[cols].copy()
    con.execute("DELETE FROM matches")
    con.register("_matches_df", df)
    con.execute("INSERT INTO matches SELECT * FROM _matches_df")
    con.unregister("_matches_df")
    return len(df)


# ---------------------------------------------------------------------------
# ILLUSTRATIVE 2026 World Cup knockout fixtures.
#
# NOT the real draw. These are placeholder pairings using canonical team_ids so
# downstream KO/bracket/API code has valid rows to run against. Replace with the
# actual bracket once the draw is wired. neutral=True everywhere except the three
# hosts get host_flag=True (spec §7.3).
# ---------------------------------------------------------------------------
_ILLUSTRATIVE_FIXTURES = [
    # (date, home_name, away_name, stage)
    ("2026-07-11", "United States", "Brazil", "Quarter-final"),
    ("2026-07-11", "France", "Argentina", "Quarter-final"),
    ("2026-07-12", "England", "Spain", "Quarter-final"),
    ("2026-07-12", "Netherlands", "Portugal", "Quarter-final"),
    ("2026-07-14", "Brazil", "France", "Semi-final"),
    ("2026-07-15", "Spain", "Netherlands", "Semi-final"),
    ("2026-07-18", "France", "Spain", "Third place"),
    ("2026-07-19", "Brazil", "Netherlands", "Final"),
]

_HOST_NAMES = {"United States", "Canada", "Mexico"}


def seed_illustrative_fixtures(con: duckdb.DuckDBPyConnection) -> int:
    """Seed the small illustrative 2026 KO fixture set. Returns row count.

    Idempotent: clears ``fixtures`` first. Team names resolved via canonical.
    """
    rows = []
    for i, (date, home, away, stage) in enumerate(_ILLUSTRATIVE_FIXTURES):
        home_id = canonical.resolve_strict(home)
        away_id = canonical.resolve_strict(away)
        host_flag = home in _HOST_NAMES  # hosts keep home advantage (spec §7.3)
        rows.append(
            {
                "fixture_id": i,
                "date": pd.to_datetime(date).date(),
                "home_id": home_id,
                "away_id": away_id,
                "stage": stage,
                "neutral": not host_flag,
                "host_flag": host_flag,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=["fixture_id", "date", "home_id", "away_id", "stage", "neutral", "host_flag"],
    )
    con.execute("DELETE FROM fixtures")
    con.register("_fix_df", df)
    con.execute("INSERT INTO fixtures SELECT * FROM _fix_df")
    con.unregister("_fix_df")
    return len(df)
