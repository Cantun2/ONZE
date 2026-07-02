"""ONZE serving API — FastAPI thin layer over predict/ and eval/ (spec §4.4, §5).

This is a **thin serving layer**. It does not contain any modelling logic and it
does not do heavy compute in the request path:

* ``GET`` endpoints read *precomputed* artifacts — the ``predictions`` table
  (matrix + 1N2, populated by :mod:`src.predict.fixture`), team/fixture metadata,
  and, for the backtest, an on-disk eval report. Deriving markets (1N2 / O/U /
  BTTS / top-5) from a stored matrix is pure summation over a small ``P(x, y)``
  (spec §3.4), which is cheap and consistent with the matrix by construction.
* The **only** endpoint allowed to trigger recompute is ``POST /refresh``
  (spec §4.4), which re-runs the predict pipeline on new results/odds.

Design choices (documented, per the charter)
-------------------------------------------
* **Predictions** are read straight from ``predictions.matrix_json`` and
  round-trip losslessly (JSON list-of-lists of Python floats, no precision loss;
  see :func:`_matrix_from_json`). If a fixture has no precomputed row we return
  ``404`` (call ``POST /refresh`` to populate it) rather than computing in the GET
  path.
* **Advance probabilities** for knockout fixtures need the KO chain
  (lambdas + Elo + extra-time + penalties), which is not stored in the
  ``predictions`` table. We obtain them via :func:`src.predict.fixture.predict_fixture`
  and **memoise per (fixture_id, model_version)** in-process. With the seeded set
  of 8 static fixtures this is effectively free and stays consistent with the
  per-fixture qualification numbers; the cache is cleared by ``POST /refresh``.
* **Bracket** runs a Monte-Carlo simulation (:func:`src.predict.bracket.simulate_tournament`).
  There is no persisted bracket artifact, so we run a **smaller, seeded** sim on
  the first call and **cache the result in memory** (cleared by ``POST /refresh``).
  Subsequent calls are served from cache — the GET path stays non-blocking.
* **Backtest** reads an on-disk eval report if the eval module has produced one.
  The eval module may not be finished yet, so we import it lazily and return a
  clean ``503`` when neither a report nor a callable is available.

CORS is enabled for the Vite dev frontend; origins are configurable via the
``ONZE_CORS_ORIGINS`` env var (comma-separated).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.predict.bracket import played_result
from src.predict.fixture import is_knockout_stage, news_delta
from src.predict.markets import derive_markets

# Window (days) used to match an already-played ``matches`` row to a knockout
# fixture's scheduled date -- mirrors ``src.predict.bracket._PLAYED_MATCH_WINDOW_DAYS``
# so the goals we surface in ``/fixtures`` agree with the winner
# ``played_result`` resolves (same tie, not an earlier meeting between the pair).
_PLAYED_MATCH_WINDOW_DAYS = 3

MODEL_VERSION = "dc-v1"

# Where the eval module is expected to persist its backtest report, if/when it
# runs. We read this artifact (never recompute the backtest in-request).
_EVAL_REPORT_CANDIDATES = ("backtest_report.json", "backtest.json", "report.json")


# ---------------------------------------------------------------------------
# Pydantic response models (typed + documented -> OpenAPI, spec DoD)
# ---------------------------------------------------------------------------
class ResultOut(BaseModel):
    """Actual result of an already-played knockout tie (spec §4.4 /fixtures).

    Goals are oriented to the *fixture's* home/away (not necessarily the
    ``matches`` row's own orientation, which can differ, e.g. neutral-venue
    ties recorded with the other team as "home").
    """

    home_goals: int
    away_goals: int
    winner_id: int
    winner_team: str
    shootout: bool = Field(
        ..., description="True when the 90'/ET score was level and the winner came from penalties."
    )


class FixtureOut(BaseModel):
    """One remaining match with metadata and joined team names (spec §4.4 /fixtures)."""

    fixture_id: int
    date: Optional[str] = Field(None, description="ISO date of the fixture.")
    home_id: int
    away_id: int
    home_team: str = Field(..., description="Canonical name of the home team.")
    away_team: str = Field(..., description="Canonical name of the away team.")
    stage: Optional[str] = Field(None, description="Tournament stage, e.g. 'Quarter-final'.")
    neutral: bool
    host_flag: bool
    knockout: bool = Field(..., description="True if this is a knockout tie (spec §3.5).")
    played: bool = Field(
        False, description="True if this tie already has a completed result in `matches`."
    )
    result: Optional[ResultOut] = Field(
        None, description="Actual result when `played` is true; null otherwise."
    )


class OverUnderLine(BaseModel):
    """Over/Under probabilities for one goals line (spec §3.4)."""

    over: float
    under: float


class ScoreProb(BaseModel):
    """A single scoreline and its probability."""

    home: int
    away: int
    prob: float


class AdvanceOut(BaseModel):
    """Knockout qualification probabilities (spec §3.5). Sum to ~1."""

    p_home_advance: float
    p_away_advance: float


class NewsSideOut(BaseModel):
    """One team's live news-adjustment overlay (subjective, prediction-time only)."""

    delta: float = Field(0.0, description="Elo delta applied at prediction time (0 if none).")
    note: Optional[str] = Field(None, description="Human-readable reason, e.g. an injury note.")


class NewsAdjustmentOut(BaseModel):
    """Home/away news overlay for a fixture (spec: never touches stored ratings)."""

    home: NewsSideOut
    away: NewsSideOut


class PredictionOut(BaseModel):
    """Full P(x,y) matrix + derived markets for one fixture (spec §4.4 /predict)."""

    fixture_id: int
    model_version: str
    home_team: str
    away_team: str
    stage: Optional[str] = None
    knockout: bool
    matrix: list[list[float]] = Field(
        ..., description="P(x,y): matrix[x][y] = P(home scores x, away scores y)."
    )
    p_home: float
    p_draw: float
    p_away: float
    most_likely_score: str
    top5_scores: list[ScoreProb]
    over_under: dict[str, OverUnderLine]
    btts: float
    advance: Optional[AdvanceOut] = Field(
        None, description="Present only for knockout fixtures (spec §3.5)."
    )
    news_adjustment: Optional[NewsAdjustmentOut] = Field(
        None,
        description=(
            "Live per-team Elo overlay from config.yaml news_adjustments "
            "(injury/suspension nudges), applied at prediction time only."
        ),
    )


class BracketTeamOut(BaseModel):
    """Per-team tournament advancement probabilities (spec §1, §5 view 3)."""

    team_id: int
    team: str
    p_reach_semi: float
    p_reach_final: float
    p_champion: float


class BracketOut(BaseModel):
    """Bracket Monte-Carlo result: per-team probabilities + sim metadata (spec §5)."""

    n_sims: int
    seed: int
    teams: list[BracketTeamOut]


class BacktestRowOut(BaseModel):
    """One model/baseline's RPS (and optionally other metrics) from the report."""

    name: str
    rps: Optional[float] = None
    log_loss: Optional[float] = None
    brier: Optional[float] = None
    n_matches: Optional[int] = None


class CalibrationBinOut(BaseModel):
    """One reliability-diagram bin (spec §3.7): predicted vs observed frequency."""

    bin_lo: float
    bin_hi: float
    p_pred: float = Field(..., description="Mean predicted probability in the bin.")
    p_obs: float = Field(..., description="Observed frequency in the bin.")
    n: int = Field(..., description="Number of predictions falling in the bin.")


class BacktestOut(BaseModel):
    """Model vs Elo (vs bookmaker) comparison from the eval report (spec §4.4).

    The first three fields (``from_year``, ``results``, ``source``) are the stable
    contract the frontend binds. The remaining fields are optional pass-throughs of
    richer artifact content (calibration curve, ECE, chosen ``xi``, honesty caveat,
    window metadata) so the frontend can draw the reliability diagram (spec §3.7,
    §5 view 4). They default to ``None`` so an older artifact lacking them still
    serialises cleanly.
    """

    from_year: Optional[int] = None
    results: list[BacktestRowOut]
    source: str = Field(..., description="Where the report was read from.")
    calibration: Optional[list[CalibrationBinOut]] = Field(
        None, description="Reliability-diagram bins (spec §3.7), if present."
    )
    ece: Optional[float] = Field(None, description="Expected calibration error.")
    xi_chosen: Optional[float] = Field(None, description="Time-decay rate chosen by backtest.")
    xi_half_life_days: Optional[float] = Field(
        None, description="Half-life (days) that ``xi_chosen`` encodes."
    )
    caveat: Optional[str] = Field(None, description="Honesty / sample-size note (spec §3.7).")
    generated_from: Optional[str] = Field(None, description="Backtest window start.")
    n_matches: Optional[int] = Field(None, description="Matches in the backtest window.")


class RefreshOut(BaseModel):
    """Result of ``POST /refresh``: recompute status + counts (spec §4.4)."""

    status: str
    predictions_written: int
    caches_cleared: bool


# ---------------------------------------------------------------------------
# In-process caches (documented above). Cleared by POST /refresh.
# ---------------------------------------------------------------------------
_BRACKET_CACHE: dict[str, Any] = {}
_ADVANCE_CACHE: dict[tuple[int, str], dict[str, float]] = {}

# Bracket sim size: smaller than the offline default (100k) to keep the first
# call responsive; seeded so it is reproducible. Result is memoised afterwards.
_BRACKET_N_SIMS = 20_000


def _db_connect(read_only: bool = True):
    """Open a DuckDB connection to the configured store."""
    import duckdb

    cfg = load_config()
    return duckdb.connect(str(cfg.paths.db_path), read_only=read_only)


def _matrix_from_json(matrix_json: str) -> list[list[float]]:
    """Round-trip the stored ``matrix_json`` to a list-of-lists of floats.

    The matrix is persisted as a JSON list-of-lists (see
    :func:`src.predict.fixture._row_for_predictions`); ``json.loads`` restores the
    exact Python floats with no precision loss.
    """
    return json.loads(matrix_json)


def _advance_for(fixture_id: int, stage: Optional[str]) -> Optional[dict[str, float]]:
    """Advance probabilities for a knockout fixture, memoised (documented choice).

    Not stored in ``predictions``; obtained from the canonical predict pipeline
    and cached per ``(fixture_id, model_version)`` so the GET path is light.
    """
    if not is_knockout_stage(stage):
        return None
    key = (fixture_id, MODEL_VERSION)
    if key not in _ADVANCE_CACHE:
        from src.predict.fixture import predict_fixture

        res = predict_fixture(fixture_id, model_version=MODEL_VERSION)
        adv = res.get("advance")
        _ADVANCE_CACHE[key] = adv if adv is not None else {}
    adv = _ADVANCE_CACHE[key]
    return adv or None


def _fixture_result(con, home_id: int, away_id: int, date: Any) -> Optional[dict[str, Any]]:
    """Actual result for an already-played knockout tie, or ``None`` if unplayed.

    Thin read: :func:`src.predict.bracket.played_result` resolves the winner
    (handling shootouts via the historical record); one extra indexed query on
    ``matches`` fetches the 90'/ET goals for the same windowed pair so the
    scoreline and the winner stay consistent. No modelling logic here.
    """
    fx_date: Optional[_dt.date] = None
    if isinstance(date, _dt.datetime):
        fx_date = date.date()
    elif isinstance(date, _dt.date):
        fx_date = date
    elif date is not None:
        try:
            fx_date = _dt.date.fromisoformat(str(date)[:10])
        except ValueError:
            fx_date = None

    winner_id = played_result(con, home_id, away_id, fx_date)
    if winner_id is None:
        return None

    try:
        if fx_date is not None:
            lo = fx_date - _dt.timedelta(days=_PLAYED_MATCH_WINDOW_DAYS)
            hi = fx_date + _dt.timedelta(days=_PLAYED_MATCH_WINDOW_DAYS)
            row = con.execute(
                """
                SELECT home_id, away_id, home_goals, away_goals
                FROM matches
                WHERE ((home_id = ? AND away_id = ?) OR (home_id = ? AND away_id = ?))
                  AND home_goals IS NOT NULL AND away_goals IS NOT NULL
                  AND date BETWEEN ? AND ?
                ORDER BY date DESC LIMIT 1
                """,
                [home_id, away_id, away_id, home_id, lo, hi],
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT home_id, away_id, home_goals, away_goals
                FROM matches
                WHERE ((home_id = ? AND away_id = ?) OR (home_id = ? AND away_id = ?))
                  AND home_goals IS NOT NULL AND away_goals IS NOT NULL
                ORDER BY date DESC LIMIT 1
                """,
                [home_id, away_id, away_id, home_id],
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None

    m_home, m_away, hg, ag = row
    # Orient goals to the *fixture's* home/away, not the matches row's own.
    if int(m_home) == int(home_id):
        home_goals, away_goals = int(hg), int(ag)
    else:
        home_goals, away_goals = int(ag), int(hg)

    winner_row = con.execute(
        "SELECT name_canonical FROM teams WHERE team_id = ?", [int(winner_id)]
    ).fetchone()
    winner_team = winner_row[0] if winner_row else str(winner_id)

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner_id": int(winner_id),
        "winner_team": winner_team,
        "shootout": bool(hg == ag),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ONZE API",
    version="1.0.0",
    description=(
        "Serving layer for ONZE — score-distribution engine for the 2026 World "
        "Cup (spec §4.4). Thin layer: reads precomputed predictions; heavy "
        "recompute only via POST /refresh."
    ),
)


def _cors_origins() -> list[str]:
    """CORS allow-list for the frontend, configurable via env (spec §4.1 Vite)."""
    raw = os.environ.get("ONZE_CORS_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Sensible defaults for the Vite dev server.
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    """Service banner + endpoint index."""
    return {
        "service": "ONZE API",
        "model_version": MODEL_VERSION,
        "endpoints": [
            "GET /fixtures",
            "GET /predict/{fixture_id}",
            "GET /bracket",
            "GET /eval/backtest?from=YYYY",
            "POST /refresh",
        ],
    }


@app.get("/fixtures", response_model=list[FixtureOut], tags=["fixtures"])
def get_fixtures() -> list[FixtureOut]:
    """Remaining matches + metadata, with team names joined (spec §4.4)."""
    con = _db_connect(read_only=True)
    try:
        rows = con.execute(
            """
            SELECT f.fixture_id, f.date, f.home_id, f.away_id, f.stage,
                   f.neutral, f.host_flag,
                   th.name_canonical AS home_team,
                   ta.name_canonical AS away_team
            FROM fixtures f
            LEFT JOIN teams th ON th.team_id = f.home_id
            LEFT JOIN teams ta ON ta.team_id = f.away_id
            ORDER BY f.date, f.fixture_id
            """
        ).fetchall()

        out: list[FixtureOut] = []
        for r in rows:
            (fid, date, home_id, away_id, stage, neutral, host_flag, home_team, away_team) = r
            # Single indexed lookup per fixture (16 rows) -- thin read, no compute.
            result = _fixture_result(con, int(home_id), int(away_id), date)
            out.append(
                FixtureOut(
                    fixture_id=int(fid),
                    date=str(date) if date is not None else None,
                    home_id=int(home_id),
                    away_id=int(away_id),
                    home_team=home_team or str(home_id),
                    away_team=away_team or str(away_id),
                    stage=stage,
                    neutral=bool(neutral),
                    host_flag=bool(host_flag),
                    knockout=is_knockout_stage(stage),
                    played=result is not None,
                    result=ResultOut(**result) if result is not None else None,
                )
            )
    finally:
        con.close()
    return out


@app.get("/predict/{fixture_id}", response_model=PredictionOut, tags=["predict"])
def get_prediction(fixture_id: int) -> PredictionOut:
    """P(x,y) matrix + derived markets for one fixture (spec §4.4, §3.4, §3.5).

    Reads the **precomputed** row from ``predictions``; if none exists returns
    404 (call ``POST /refresh`` to populate it). Markets are re-summed from the
    stored matrix (cheap, consistent). Knockout fixtures also carry qualification
    probabilities (spec §3.5), memoised.
    """
    con = _db_connect(read_only=True)
    try:
        row = con.execute(
            """
            SELECT p.matrix_json, p.model_version, f.stage, f.home_id, f.away_id,
                   th.name_canonical AS home_team,
                   ta.name_canonical AS away_team
            FROM predictions p
            JOIN fixtures f ON f.fixture_id = p.fixture_id
            LEFT JOIN teams th ON th.team_id = f.home_id
            LEFT JOIN teams ta ON ta.team_id = f.away_id
            WHERE p.fixture_id = ? AND p.model_version = ?
            ORDER BY p.created_at DESC
            LIMIT 1
            """,
            [int(fixture_id), MODEL_VERSION],
        ).fetchone()
        fixture_exists = con.execute(
            "SELECT 1 FROM fixtures WHERE fixture_id = ?", [int(fixture_id)]
        ).fetchone()
    finally:
        con.close()

    if row is None:
        if fixture_exists is None:
            raise HTTPException(status_code=404, detail=f"unknown fixture_id={fixture_id}")
        raise HTTPException(
            status_code=404,
            detail=(
                f"no precomputed prediction for fixture_id={fixture_id} "
                f"(model_version={MODEL_VERSION}); call POST /refresh"
            ),
        )

    matrix_json, model_version, stage, home_id, away_id, home_team, away_team = row
    matrix = _matrix_from_json(matrix_json)

    # Derived markets are pure sums over the stored matrix (spec §3.4) — light.
    markets = derive_markets(matrix)

    top5 = [
        ScoreProb(home=int(xy[0]), away=int(xy[1]), prob=float(p))
        for (xy, p) in markets["top5_scores"]
    ]
    over_under = {
        line: OverUnderLine(over=float(v["over"]), under=float(v["under"]))
        for line, v in markets["over_under"].items()
    }

    adv_dict = _advance_for(int(fixture_id), stage)
    advance = (
        AdvanceOut(
            p_home_advance=float(adv_dict["p_home_advance"]),
            p_away_advance=float(adv_dict["p_away_advance"]),
        )
        if adv_dict
        else None
    )

    # Live news overlay (spec: prediction-time only, never persisted). Cheap
    # config lookup per side -- no full predict_fixture recompute in the GET path.
    cfg = load_config()
    home_delta, home_note = news_delta(int(home_id), cfg)
    away_delta, away_note = news_delta(int(away_id), cfg)
    news_adjustment = NewsAdjustmentOut(
        home=NewsSideOut(delta=float(home_delta), note=home_note if home_delta else None),
        away=NewsSideOut(delta=float(away_delta), note=away_note if away_delta else None),
    )

    return PredictionOut(
        fixture_id=int(fixture_id),
        model_version=model_version,
        home_team=home_team or str(fixture_id),
        away_team=away_team or str(fixture_id),
        stage=stage,
        knockout=is_knockout_stage(stage),
        matrix=matrix,
        p_home=float(markets["p_home"]),
        p_draw=float(markets["p_draw"]),
        p_away=float(markets["p_away"]),
        most_likely_score=markets["most_likely_score_str"],
        top5_scores=top5,
        over_under=over_under,
        btts=float(markets["btts"]),
        advance=advance,
        news_adjustment=news_adjustment,
    )


@app.get("/bracket", response_model=BracketOut, tags=["bracket"])
def get_bracket() -> BracketOut:
    """Per-team advancement/champion probabilities (spec §1, §5 view 3).

    No persisted bracket artifact exists, so on the first call we run a smaller,
    seeded Monte-Carlo sim and cache the result in memory (cleared by
    ``POST /refresh``). Subsequent calls are served from cache — GET stays light.
    """
    if not _BRACKET_CACHE:
        try:
            from src.predict.bracket import DEFAULT_SEED, simulate_tournament

            df = simulate_tournament(n_sims=_BRACKET_N_SIMS, seed=DEFAULT_SEED)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=503,
                detail=f"bracket simulation unavailable: {exc}",
            )
        _BRACKET_CACHE["teams"] = df.to_dict("records")
        _BRACKET_CACHE["n_sims"] = int(df.attrs.get("n_sims", _BRACKET_N_SIMS))
        _BRACKET_CACHE["seed"] = int(df.attrs.get("seed", DEFAULT_SEED))

    teams = [
        BracketTeamOut(
            team_id=int(r["team_id"]),
            team=str(r["team"]),
            p_reach_semi=float(r["p_reach_semi"]),
            p_reach_final=float(r["p_reach_final"]),
            p_champion=float(r["p_champion"]),
        )
        for r in _BRACKET_CACHE["teams"]
    ]
    return BracketOut(
        n_sims=_BRACKET_CACHE["n_sims"],
        seed=_BRACKET_CACHE["seed"],
        teams=teams,
    )


def _load_eval_report() -> Optional[dict[str, Any]]:
    """Read a persisted backtest report from disk, if the eval module wrote one.

    Looks in the processed data dir for a known report filename. Returns the
    parsed dict or ``None`` if no artifact exists.
    """
    cfg = load_config()
    search_dirs = [cfg.paths.processed_dir, cfg.paths.processed_dir / "eval"]
    for d in search_dirs:
        for name in _EVAL_REPORT_CANDIDATES:
            p: Path = d / name
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
    return None


def _as_float(v: Any) -> Optional[float]:
    """Coerce ``v`` to float, or ``None`` if absent/non-numeric."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> Optional[int]:
    """Coerce ``v`` to int, or ``None`` if absent/non-numeric."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_calibration(raw: Any) -> Optional[list[CalibrationBinOut]]:
    """Parse the calibration bins from the report, tolerating malformed entries.

    Expects a list of ``{bin_lo, bin_hi, p_pred, p_obs, n}`` dicts (spec §3.7).
    Returns ``None`` when absent; skips any bin missing required numeric fields so
    a partially-malformed artifact still yields a usable curve.
    """
    if not isinstance(raw, list) or not raw:
        return None
    bins: list[CalibrationBinOut] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lo = _as_float(item.get("bin_lo"))
        hi = _as_float(item.get("bin_hi"))
        p_pred = _as_float(item.get("p_pred"))
        p_obs = _as_float(item.get("p_obs"))
        n = _as_int(item.get("n"))
        if None in (lo, hi, p_pred, p_obs, n):
            continue
        bins.append(
            CalibrationBinOut(bin_lo=lo, bin_hi=hi, p_pred=p_pred, p_obs=p_obs, n=n)
        )
    return bins or None


def _normalise_backtest_report(report: dict[str, Any]) -> list[BacktestRowOut]:
    """Coerce a report dict into typed rows, tolerating a few plausible shapes."""
    rows: list[BacktestRowOut] = []

    def _row(name: str, d: dict[str, Any]) -> BacktestRowOut:
        return BacktestRowOut(
            name=name,
            rps=d.get("rps") if isinstance(d, dict) else None,
            log_loss=d.get("log_loss") if isinstance(d, dict) else None,
            brier=d.get("brier") if isinstance(d, dict) else None,
            n_matches=d.get("n_matches") if isinstance(d, dict) else None,
        )

    if isinstance(report.get("results"), list):
        for item in report["results"]:
            if isinstance(item, dict):
                rows.append(_row(str(item.get("name", "model")), item))
    elif isinstance(report.get("models"), dict):
        for name, d in report["models"].items():
            rows.append(_row(str(name), d if isinstance(d, dict) else {"rps": d}))
    else:
        # Flat mapping of name -> rps (or name -> metrics dict).
        for name, d in report.items():
            if name in ("from", "from_year"):
                continue
            if isinstance(d, dict):
                rows.append(_row(str(name), d))
            elif isinstance(d, (int, float)):
                rows.append(BacktestRowOut(name=str(name), rps=float(d)))
    return rows


@app.get("/eval/backtest", response_model=BacktestOut, tags=["eval"])
def get_backtest(
    from_year: Optional[int] = Query(
        None, alias="from", description="Only include matches from this year onward."
    ),
) -> BacktestOut:
    """Model vs Elo (vs bookmaker) RPS from the eval report (spec §4.4, §3.7).

    Reads a persisted eval **report** — it never recomputes the walk-forward
    backtest in-request. If the eval module has not produced a report yet, tries
    a lazily-imported report-loader, and otherwise returns a clean ``503``.
    """
    # 1) Preferred: an on-disk report artifact.
    report = _load_eval_report()
    source = "eval report artifact"

    # 2) Fallback: a callable exposed by the eval module (imported lazily so a
    #    not-yet-finished eval module cannot break app import).
    if report is None:
        try:
            from src.eval import backtest as backtest_mod  # type: ignore
        except Exception:
            backtest_mod = None  # eval module not present yet
        if backtest_mod is not None:
            for fn_name in ("load_report", "latest_report", "get_report"):
                fn = getattr(backtest_mod, fn_name, None)
                if callable(fn):
                    try:
                        report = fn()
                        source = f"src.eval.backtest.{fn_name}()"
                        break
                    except Exception:
                        report = None

    if not report:
        raise HTTPException(
            status_code=503,
            detail=(
                "backtest results not available yet — the eval module has not "
                "produced a report. Run the eval backtest first."
            ),
        )

    rows = _normalise_backtest_report(report)
    if not rows:
        raise HTTPException(
            status_code=503,
            detail="backtest report present but contained no parseable results.",
        )

    reported_from = report.get("from_year") or report.get("from")

    # Optional pass-throughs for the reliability curve / honesty panel (spec §3.7,
    # §5 view 4). All defaulted so an older artifact lacking them still serialises.
    calibration = _parse_calibration(report.get("calibration"))

    return BacktestOut(
        from_year=from_year if from_year is not None else reported_from,
        results=rows,
        source=source,
        calibration=calibration,
        ece=_as_float(report.get("ece")),
        xi_chosen=_as_float(report.get("xi_chosen")),
        xi_half_life_days=_as_float(report.get("xi_half_life_days")),
        caveat=report.get("caveat") if isinstance(report.get("caveat"), str) else None,
        generated_from=(
            report.get("generated_from")
            if isinstance(report.get("generated_from"), str)
            else None
        ),
        n_matches=_as_int(report.get("n_matches")),
    )


@app.post("/refresh", response_model=RefreshOut, tags=["refresh"])
def post_refresh() -> RefreshOut:
    """Recompute predictions on new results/odds (spec §4.4).

    The **only** endpoint allowed heavy compute: it re-runs the predict pipeline
    (:func:`src.predict.fixture.main`), which repopulates the ``predictions``
    table, and clears the in-process bracket/advance caches so subsequent GETs
    reflect the fresh predictions.
    """
    try:
        from src.predict.fixture import main as predict_main

        predict_main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"refresh failed: {exc}")

    # Count what we now have and clear derived caches.
    con = _db_connect(read_only=True)
    try:
        n = con.execute(
            "SELECT count(*) FROM predictions WHERE model_version = ?",
            [MODEL_VERSION],
        ).fetchone()[0]
    finally:
        con.close()

    _BRACKET_CACHE.clear()
    _ADVANCE_CACHE.clear()

    return RefreshOut(status="ok", predictions_written=int(n), caches_cleared=True)
