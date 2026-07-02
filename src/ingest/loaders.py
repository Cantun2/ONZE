"""Dataset loaders — download raw sources, clean into typed Parquet + DuckDB.

Pipeline (spec §2.1, §2.2, phase 0):

1. Download the mandatory martj42/international_results CSVs to ``data/raw``
   (immutable). Best-effort fetch of the Fjelstul worldcup enrichment CSVs.
2. Clean martj42 ``results.csv`` into a typed Parquet in ``data/processed``:
   resolve both team names to canonical ``team_id`` (spec §2.3 gate), map each
   ``tournament`` string to an ``importance_k`` (config ``k_by_tournament``),
   carry the ``neutral`` flag, and derive ``host_flag`` (team plays in its own
   country and the match is not on neutral ground — spec §3.1/§7 hosts concept).
3. Build the DuckDB schema (spec §2.2) and populate ``teams`` + ``matches``.
   ``ratings_history`` / ``predictions`` / ``odds`` are left empty (owned
   downstream); a small ILLUSTRATIVE set of 2026 KO fixtures is seeded.

Run the whole thing with::

    python -m src.ingest.loaders

Never fabricates match data: a missing score becomes SQL NULL, documented.
Raw CSVs are downloaded once and never mutated; cleaning writes copies only.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.config import Config, load_config
from src.ingest import canonical
from src.ingest.schema import (
    create_schema,
    populate_matches,
    populate_teams,
    seed_illustrative_fixtures,
)

# ---------------------------------------------------------------------------
# Source URLs (spec §2.1).
# ---------------------------------------------------------------------------
MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
MARTJ42_FILES = {
    "martj42_results.csv": f"{MARTJ42_BASE}/results.csv",
    "martj42_shootouts.csv": f"{MARTJ42_BASE}/shootouts.csv",
}

FJELSTUL_BASE = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
FJELSTUL_FILES = {
    "fjelstul_matches.csv": f"{FJELSTUL_BASE}/matches.csv",
    "fjelstul_teams.csv": f"{FJELSTUL_BASE}/teams.csv",
    "fjelstul_tournaments.csv": f"{FJELSTUL_BASE}/tournaments.csv",
}

# 2026 World Cup hosts keep a home-field advantage despite neutral venues
# (spec §7.3). Their canonical names.
HOSTS_2026 = ("United States", "Canada", "Mexico")

_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Download (raw, immutable)
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, *, force: bool = False) -> bool:
    """Fetch *url* to *dest* unless it already exists. Return True on success."""
    if dest.exists() and not force:
        return True
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:  # network / HTTP error
        print(f"  ! failed to fetch {url}: {exc}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def download_sources(cfg: Config, *, force: bool = False) -> dict[str, bool]:
    """Download every source into ``raw_dir``. Return per-file success flags.

    martj42 is mandatory; Fjelstul is best-effort (a failure is logged, not
    fatal). Transfermarkt squad values are Kaggle-gated and skipped by design
    (documented as absent — spec §2.1).
    """
    raw = cfg.paths.raw_dir
    status: dict[str, bool] = {}
    print("Downloading martj42 (mandatory)...")
    for name, url in MARTJ42_FILES.items():
        ok = _download(url, raw / name, force=force)
        status[name] = ok
        print(f"  {name}: {'ok' if ok else 'FAILED'}")
    print("Downloading Fjelstul worldcup (best-effort)...")
    for name, url in FJELSTUL_FILES.items():
        ok = _download(url, raw / name, force=force)
        status[name] = ok
        print(f"  {name}: {'ok' if ok else 'skipped/unavailable'}")
    # Mandatory source must be present.
    for mand in MARTJ42_FILES:
        if not status.get(mand):
            raise RuntimeError(
                f"Mandatory source {mand} could not be downloaded — cannot build."
            )
    return status


# ---------------------------------------------------------------------------
# Tournament -> importance_k mapping (spec §3.1, config k_by_tournament)
# ---------------------------------------------------------------------------
def map_importance_k(tournament: str, k_by_tournament: dict[str, float]) -> float:
    """Map a raw ``tournament`` string to its Elo importance weight ``K``.

    Rules (spec §3.1):
      * "FIFA World Cup" (the finals)   -> world_cup_finals
      * anything ending in "qualification" -> qualifier
      * "Friendly"                       -> friendly
      * everything else                  -> default
    """
    default = k_by_tournament["default"]
    if tournament is None:
        return default
    t = tournament.strip()
    tl = t.lower()
    if tl == "friendly":
        return k_by_tournament["friendly"]
    if tl.endswith("qualification"):
        return k_by_tournament["qualifier"]
    if t == "FIFA World Cup":
        return k_by_tournament["world_cup_finals"]
    return default


# ---------------------------------------------------------------------------
# Cleaning: martj42 results -> typed matches frame
# ---------------------------------------------------------------------------
def clean_matches(cfg: Config) -> pd.DataFrame:
    """Read raw martj42 results.csv, resolve names, type, derive columns.

    Returns a frame with columns:
      match_id, date, home_team, away_team, home_id, away_id,
      home_goals, away_goals, neutral, tournament, importance_k, host_flag
    Rows whose team names are unresolved are a HARD ERROR (gate, spec §2.3).
    Missing scores stay as NA (never fabricated).
    """
    raw_path = cfg.paths.raw_dir / "martj42_results.csv"
    df = pd.read_csv(raw_path, dtype={"home_team": "string", "away_team": "string"})

    # --- gate: every name must resolve -------------------------------------
    names = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    unresolved = canonical.unresolved_names([n for n in names if pd.notna(n)])
    if unresolved:
        raise RuntimeError(
            "Name-harmonisation gate FAILED — unresolved team names: "
            + ", ".join(unresolved)
        )

    # --- typing -------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # scores: nullable integers, NA preserved (never fabricated)
    df["home_goals"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    # neutral flag: martj42 stores TRUE/FALSE strings
    df["neutral"] = (
        df["neutral"].astype("string").str.strip().str.upper().eq("TRUE")
    )

    # --- canonical ids ------------------------------------------------------
    df["home_id"] = df["home_team"].map(canonical.resolve).astype("Int64")
    df["away_id"] = df["away_team"].map(canonical.resolve).astype("Int64")

    # --- importance_k -------------------------------------------------------
    df["tournament"] = df["tournament"].astype("string")
    df["importance_k"] = df["tournament"].map(
        lambda t: map_importance_k(t, cfg.k_by_tournament)
    )

    # --- host_flag: home team plays in its own country and not neutral ------
    # (spec §3.1/§7). martj42 'country' is where the match is played.
    country = df["country"].astype("string")
    df["host_flag"] = (~df["neutral"]) & (df["home_team"] == country)

    # stable chronological match_id
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    df["match_id"] = df.index.astype("int64")

    cols = [
        "match_id", "date", "home_team", "away_team", "home_id", "away_id",
        "home_goals", "away_goals", "neutral", "tournament", "importance_k",
        "host_flag",
    ]
    return df[cols]


def write_processed_parquet(cfg: Config, matches: pd.DataFrame) -> Path:
    """Write the cleaned matches frame to processed Parquet. Return its path."""
    out_dir = cfg.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "matches.parquet"
    matches.to_parquet(out, index=False)
    # also persist the canonical team table for inspection / downstream reuse
    teams = pd.DataFrame(canonical.build_team_table())
    teams.to_parquet(out_dir / "teams.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Full ingest: download -> clean -> Parquet -> DuckDB schema + populate."""
    cfg = load_config()
    print("=== ONZE ingest ===")
    status = download_sources(cfg)

    print("Cleaning martj42 matches...")
    matches = clean_matches(cfg)
    n = len(matches)
    dmin, dmax = matches["date"].min(), matches["date"].max()
    n_missing = int(matches["home_goals"].isna().sum() + matches["away_goals"].isna().sum())
    print(f"  {n} matches, {dmin} -> {dmax}, {n_missing} missing score cells (kept NULL)")

    parquet = write_processed_parquet(cfg, matches)
    print(f"  wrote {parquet}")

    print("Building DuckDB schema + populating teams/matches...")
    con = create_schema(cfg)
    try:
        n_teams = populate_teams(con, canonical.build_team_table())
        n_matches = populate_matches(con, matches)
        n_fix = seed_illustrative_fixtures(con)
        con.commit()
    finally:
        con.close()
    print(f"  teams={n_teams}  matches={n_matches}  fixtures(illustrative)={n_fix}")

    # --- ingest report ------------------------------------------------------
    write_report(cfg, matches, status, n_fix)
    print("Ingest complete.")


def write_report(
    cfg: Config,
    matches: pd.DataFrame,
    status: dict[str, bool],
    n_fixtures: int,
) -> Path:
    """Write data/processed/INGEST_REPORT.md (spec deliverable §6)."""
    teams = canonical.build_team_table()
    all_names = pd.unique(
        pd.concat([matches["home_team"], matches["away_team"]], ignore_index=True)
    )
    unresolved = canonical.unresolved_names([n for n in all_names if pd.notna(n)])
    dmin, dmax = matches["date"].min(), matches["date"].max()
    n_missing = int(
        matches["home_goals"].isna().sum() + matches["away_goals"].isna().sum()
    )
    n_hosts = int(matches["host_flag"].sum())

    def src_line(name: str, mandatory: bool) -> str:
        ok = status.get(name)
        tag = "loaded" if ok else ("MISSING" if mandatory else "skipped (unavailable)")
        return f"| `{name}` | {'mandatory' if mandatory else 'best-effort'} | {tag} |"

    k_counts = matches["importance_k"].value_counts().sort_index()
    k_lines = "\n".join(
        f"| {int(k)} | {int(v)} |" for k, v in k_counts.items()
    )

    lines = f"""# ONZE — Ingest Report

Generated by `python -m src.ingest.loaders`. Phase 0 (spec §6): clean Parquet +
canonical team table + DuckDB schema.

## Name-harmonisation gate (spec §2.3, §7.1)

**Unresolved team names: {len(unresolved)}** (MUST be 0).

{"All martj42 names resolve to a canonical team_id." if not unresolved else "UNRESOLVED: " + ", ".join(unresolved)}

Canonical teams registered: **{len(teams)}**.

## martj42/international_results (training base)

- Matches loaded: **{len(matches):,}**
- Date range: **{dmin} → {dmax}**
- Missing score cells kept as NULL (never fabricated): **{n_missing}**
- Rows flagged `host_flag` (home team in own country, non-neutral): **{n_hosts:,}**

### importance_k distribution (spec §3.1)

| importance_k | matches |
|---|---|
{k_lines}

## Per-source status (spec §2.1)

| source | role | status |
|---|---|---|
{src_line("martj42_results.csv", True)}
{src_line("martj42_shootouts.csv", True)}
{src_line("fjelstul_matches.csv", False)}
{src_line("fjelstul_teams.csv", False)}
{src_line("fjelstul_tournaments.csv", False)}
| Transfermarkt squad values | optional proxy | skipped — Kaggle-gated, absent/NULL |

## DuckDB store (spec §2.2)

Path: `{cfg.paths.db_path}`

Populated: `teams`, `matches`. Seeded (ILLUSTRATIVE placeholders): `fixtures`
({n_fixtures} rows of 2026 KO fixtures with canonical team_ids — replace with the
real draw before production). Left empty (owned downstream): `ratings_history`,
`predictions`, `odds`.
"""
    out = cfg.paths.processed_dir / "INGEST_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lines, encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
