"""Shared pytest fixtures for ONZE.

Owned by the data-engineer: this exposes a small, self-contained **sample DB**
that downstream agents (ratings, model, predict, eval, api) can use in their own
tests without touching the network or the full 49k-row store.

Fixtures
--------
``sample_db_path``   Path to a temp DuckDB file with the §2.2 schema populated
                     with a handful of teams + matches with KNOWN values.
``sample_con``       An open DuckDB connection to that same store.
``sample_teams``     The list of team dicts used to build the sample DB.
``sample_matches``   A pandas DataFrame of the sample matches (known values).

The sample values are deliberately tiny and hand-checkable so that Elo / lambda
/ Dixon-Coles tests can assert exact numbers against them.
"""

from __future__ import annotations

import datetime as _dt

import duckdb
import pandas as pd
import pytest

from src.ingest.schema import _DDL

# --- known sample teams (ids are local to the sample, not the real registry) --
_SAMPLE_TEAMS = [
    {"team_id": 1, "name_canonical": "France", "confederation": "UEFA", "elo_current": 2000.0},
    {"team_id": 2, "name_canonical": "Brazil", "confederation": "CONMEBOL", "elo_current": 1950.0},
    {"team_id": 3, "name_canonical": "Japan", "confederation": "AFC", "elo_current": 1700.0},
    {"team_id": 4, "name_canonical": "Ghana", "confederation": "CAF", "elo_current": 1600.0},
]

# --- known sample matches (chronological, hand-checkable scores) --------------
_SAMPLE_MATCHES = [
    # match_id, date, home_id, away_id, home_goals, away_goals, neutral, tournament, importance_k
    (1, "2018-06-16", 1, 2, 2, 1, False, "FIFA World Cup", 60.0),
    (2, "2018-06-20", 3, 4, 1, 1, True, "FIFA World Cup", 60.0),
    (3, "2019-03-22", 1, 3, 3, 0, False, "Friendly", 10.0),
    (4, "2019-09-05", 2, 4, 2, 0, True, "FIFA World Cup qualification", 25.0),
    (5, "2020-11-14", 4, 1, 0, 0, False, "Friendly", 10.0),  # a draw
    (6, "2021-06-02", 2, 3, 4, 2, True, "Friendly", 10.0),
]

_SAMPLE_COLS = [
    "match_id", "date", "home_id", "away_id", "home_goals", "away_goals",
    "neutral", "tournament", "importance_k",
]


@pytest.fixture(scope="session")
def sample_teams() -> list[dict]:
    """The known sample team rows."""
    return [dict(t) for t in _SAMPLE_TEAMS]


@pytest.fixture(scope="session")
def sample_matches() -> pd.DataFrame:
    """The known sample matches as a typed DataFrame."""
    df = pd.DataFrame(_SAMPLE_MATCHES, columns=_SAMPLE_COLS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


@pytest.fixture(scope="session")
def sample_db_path(tmp_path_factory, sample_teams, sample_matches):
    """Build a temp DuckDB store with the §2.2 schema + known sample rows.

    Session-scoped so it is built once per test run. Returns the Path.
    """
    db = tmp_path_factory.mktemp("onze_sample_db") / "sample.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.execute(_DDL)
        con.register("_t", pd.DataFrame(sample_teams))
        con.execute("INSERT INTO teams SELECT * FROM _t")
        con.unregister("_t")
        con.register("_m", sample_matches[_SAMPLE_COLS])
        con.execute("INSERT INTO matches SELECT * FROM _m")
        con.unregister("_m")
        # one illustrative fixture so api/bracket tests have a row
        con.execute(
            "INSERT INTO fixtures VALUES (1, ?, 1, 2, 'Final', TRUE, FALSE)",
            [_dt.date(2026, 7, 19)],
        )
        con.commit()
    finally:
        con.close()
    return db


@pytest.fixture()
def sample_con(sample_db_path):
    """An open read/write connection to the sample DB (function-scoped)."""
    con = duckdb.connect(str(sample_db_path))
    yield con
    con.close()
