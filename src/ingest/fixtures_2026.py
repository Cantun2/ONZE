"""Seed the ``fixtures`` table with the real 2026 World Cup Round of 16.

This replaces the earlier *illustrative* placeholder fixtures with the actual
remaining knockout bracket as reported around 2026-07-01 (the tournament is
mid-flow: the Round of 32 is finishing, the Round of 16 runs 4-7 July).

Data provenance (official / press bracket, cross-checked):
  - Wikipedia "2026 FIFA World Cup knockout stage"
  - ESPN / CBS Sports 2026 World Cup bracket pages

Top half of the bracket is fully resolved by played Round-of-32 results:
    Canada 1-0 South Africa, Brazil 2-1 Japan, Paraguay 1-1 (4-3p) Germany,
    Morocco 1-1 (3-2p) Netherlands, Norway 2-1 Ivory Coast, France 3-0 Sweden,
    Mexico 2-0 Ecuador, England 2-1 DR Congo.

Bottom half still has Round-of-32 ties in progress on the seed date, so the
qualifier for those slots is **projected** as the higher current-Elo side
(``teams.elo_current``). These are flagged ``projected=True`` below and printed
in the seed report so the assumption is explicit and honest.

Venue/host handling (spec §3.1, §7.3): all 2026 matches are on neutral ground
except when a host nation (USA / Canada / Mexico) plays at home, which keeps the
home-field Elo bonus. Mexico play in Mexico City and the USA in Seattle here, so
those two ties are non-neutral with ``host_flag=True`` and the host set as the
home side; every other tie is ``neutral=True``.

Bracket adjacency (used by the Monte-Carlo sim) follows the listed order:
consecutive Round-of-16 fixtures feed the same quarter-final
(R16[0] & R16[1] -> QF0, R16[2] & R16[3] -> QF1, ...). Documented assumption.
"""

from __future__ import annotations

import datetime as _dt

import duckdb

from src.config import load_config
from src.ingest.canonical import resolve

# fixture_id, date, home team, away team, stage, neutral, host_flag, projected
# `home` carries the home-field bonus only when neutral is False (host at home).
_R16 = [
    (0, "2026-07-04", "Morocco", "Canada", "Round of 16", True, False, False),
    (1, "2026-07-04", "France", "Paraguay", "Round of 16", True, False, False),
    (2, "2026-07-05", "Brazil", "Norway", "Round of 16", True, False, False),
    (3, "2026-07-05", "Mexico", "England", "Round of 16", False, True, False),   # Mexico home, Mexico City
    (4, "2026-07-06", "Spain", "Portugal", "Round of 16", True, False, True),    # Spain proj. over Austria; Portugal proj. over Croatia
    (5, "2026-07-06", "United States", "Belgium", "Round of 16", False, True, True),  # USA home, Seattle; Belgium proj. over Senegal
    (6, "2026-07-07", "Argentina", "Australia", "Round of 16", True, False, True),    # Australia proj. over Egypt
    (7, "2026-07-07", "Switzerland", "Colombia", "Round of 16", True, False, True),   # Switzerland proj. over Algeria; Colombia proj. over Ghana
]


def seed_fixtures(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Replace all rows of ``fixtures`` with the real 2026 Round of 16.

    Returns the seeded rows as ``(fixture_id, date, home_id, away_id, stage,
    neutral, host_flag)`` tuples.
    """
    rows = []
    for fid, date, home, away, stage, neutral, host_flag, _proj in _R16:
        home_id, away_id = resolve(home), resolve(away)
        if home_id is None or away_id is None:  # pragma: no cover - names are canonical
            raise ValueError(f"Unresolved team in fixture {fid}: {home!r} / {away!r}")
        rows.append(
            (
                fid,
                _dt.date.fromisoformat(date),
                home_id,
                away_id,
                stage,
                neutral,
                host_flag,
            )
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
        names = {
            r[0]: r[1]
            for r in con.execute("SELECT team_id, name_canonical FROM teams").fetchall()
        }
        elos = {
            r[0]: r[1]
            for r in con.execute("SELECT team_id, elo_current FROM teams").fetchall()
        }
        print(f"[fixtures_2026] seeded {len(rows)} Round-of-16 fixtures:\n")
        for (fid, date, hid, aid, stage, neutral, host), meta in zip(rows, _R16):
            proj = "  (projected qualifier)" if meta[7] else ""
            venue = "neutral" if neutral else f"{names[hid]} HOME (host)"
            print(
                f"  #{fid} {date}  {names[hid]:14} ({elos[hid]:.0f}) "
                f"vs {names[aid]:14} ({elos[aid]:.0f})  [{venue}]{proj}"
            )
    finally:
        con.close()


if __name__ == "__main__":
    main()
