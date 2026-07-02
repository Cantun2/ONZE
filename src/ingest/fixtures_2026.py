"""Seed the ``fixtures`` table with the real 2026 World Cup Round of 32.

This replaces the illustrative placeholder fixtures with the actual knockout
bracket as reported around 2026-07-01. On the seed date the Round of 32 (French:
*16es de finale*) is in progress: the upper half (June 28-30) is played, and the
lower half runs 1-3 July. All 16 ties are seeded so the app shows every match
and the Monte-Carlo sim runs the full tree Round-of-32 -> Round-of-16 ->
quarter-final -> semi-final -> final.

Data provenance (official / press bracket, cross-checked):
  - Wikipedia "2026 FIFA World Cup knockout stage"
  - ESPN / CBS Sports 2026 World Cup bracket pages

Played upper-half results (for reference; the model still shows its *pre-match*
view of every tie, as the fixtures carry no result column):
    Canada 1-0 South Africa, Brazil 2-1 Japan, Paraguay 1-1 (4-3p) Germany,
    Morocco 1-1 (3-2p) Netherlands, Norway 2-1 Ivory Coast, France 3-0 Sweden.
Lower half (1-3 July): Mexico-Ecuador, USA-Bosnia, Belgium-Senegal,
England-DR Congo, Switzerland-Algeria, Portugal-Croatia, Spain-Austria,
Argentina-Cape Verde, Australia-Egypt, Colombia-Ghana.

Bracket order (``fixture_id``) IS the bracket slot. Consecutive ties feed the
same next-round tie: winners of slots ``2j`` and ``2j+1`` meet in the Round of 16
(so slots 0&1 -> Canada/SA winner vs Morocco/Netherlands winner = the "Canada vs
Morocco" R16 tie, etc.), and this repeats up to the final. The Monte-Carlo sim
(:mod:`src.predict.bracket`) reads the base round by ``fixture_id`` for exactly
this reason. Note the dates span several days, so ordering must be by slot, not
date.

Venue/host handling (spec §3.1, §7.3): all matches neutral except when a host
nation (USA / Canada / Mexico) plays at home. Here Mexico (vs Ecuador) and the
USA (vs Bosnia) play home ties and keep the Elo home bonus; Canada's tie is at a
US venue, so it is neutral. Every other tie is neutral.
"""

from __future__ import annotations

import datetime as _dt

import duckdb

from src.config import load_config
from src.ingest.canonical import resolve

# slot, date, home, away, stage, neutral, host_flag
# Order is the bracket slot; adjacent pairs (0&1, 2&3, ...) feed one R16 tie.
# `home` carries the home bonus only when neutral is False (host at home).
_R32 = [
    (0,  "2026-06-28", "Canada", "South Africa", True,  False),   # -> R16 slot A vs slot B
    (1,  "2026-06-29", "Morocco", "Netherlands", True,  False),
    (2,  "2026-06-29", "Paraguay", "Germany", True,  False),
    (3,  "2026-06-30", "France", "Sweden", True,  False),
    (4,  "2026-06-29", "Brazil", "Japan", True,  False),
    (5,  "2026-06-30", "Norway", "Ivory Coast", True,  False),
    (6,  "2026-07-01", "Mexico", "Ecuador", False, True),          # Mexico host, home
    (7,  "2026-07-01", "England", "DR Congo", True,  False),
    (8,  "2026-07-02", "Spain", "Austria", True,  False),
    (9,  "2026-07-02", "Portugal", "Croatia", True,  False),
    (10, "2026-07-01", "United States", "Bosnia and Herzegovina", False, True),  # USA host, home
    (11, "2026-07-01", "Belgium", "Senegal", True,  False),
    (12, "2026-07-03", "Argentina", "Cape Verde", True,  False),
    (13, "2026-07-03", "Australia", "Egypt", True,  False),
    (14, "2026-07-02", "Switzerland", "Algeria", True,  False),
    (15, "2026-07-03", "Colombia", "Ghana", True,  False),
]

_STAGE = "Round of 32"


def seed_fixtures(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Replace all rows of ``fixtures`` with the real 2026 Round of 32.

    Returns the seeded rows as ``(fixture_id, date, home_id, away_id, stage,
    neutral, host_flag)`` tuples.
    """
    rows = []
    for fid, date, home, away, neutral, host_flag in _R32:
        home_id, away_id = resolve(home), resolve(away)
        if home_id is None or away_id is None:  # pragma: no cover - names are canonical
            raise ValueError(f"Unresolved team in fixture {fid}: {home!r} / {away!r}")
        rows.append(
            (fid, _dt.date.fromisoformat(date), home_id, away_id, _STAGE, neutral, host_flag)
        )

    con.execute("DELETE FROM fixtures")
    con.executemany(
        "INSERT INTO fixtures "
        "(fixture_id, date, home_id, away_id, stage, neutral, host_flag) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return rows


def main() -> None:
    cfg = load_config()
    con = duckdb.connect(str(cfg.paths.db_path))
    try:
        rows = seed_fixtures(con)
        names = dict(con.execute("SELECT team_id, name_canonical FROM teams").fetchall())
        elos = dict(con.execute("SELECT team_id, elo_current FROM teams").fetchall())
        print(f"[fixtures_2026] seeded {len(rows)} Round-of-32 fixtures:\n")
        for fid, date, hid, aid, stage, neutral, host in rows:
            venue = "neutral" if neutral else f"{names[hid]} HOME (host)"
            print(
                f"  #{fid:2d} {date}  {names[hid]:14} ({elos[hid]:.0f}) "
                f"vs {names[aid]:22} ({elos[aid]:.0f})  [{venue}]"
            )
    finally:
        con.close()


if __name__ == "__main__":
    main()
