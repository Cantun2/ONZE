"""Full prediction pipeline for a single fixture (spec §1, §3.4, §3.5).

This is the orchestration layer of ONZE's ``predict`` package. For one fixture it
wires together — without re-implementing — the three upstream primitives:

1. **Elo** (:mod:`src.ratings.elo`) — both teams' current strength from
   ``teams.elo_current``, with the effective home bonus ``H_eff`` chosen from the
   fixture's ``neutral`` / ``host_flag`` exactly as the Elo engine does (spec
   §3.1, §7.3: a 2026 host keeps its home advantage even at a "neutral" venue).
2. **Goals model** (:mod:`src.model.lambdas` + :mod:`src.model.dixon_coles`) —
   ``predict_lambdas`` maps the Elo differential to ``(lambda, mu)``; then
   ``score_matrix`` builds the single object ``P(x, y)`` with the Dixon-Coles
   low-score correction (spec §3.2/§3.3).
3. **Derived markets** (:mod:`src.predict.markets`) — 1N2, top-5 scores,
   over/under, BTTS, all summed from ``P`` (spec §3.4). For a knockout fixture we
   additionally attach the qualification probabilities from
   :func:`src.model.knockout.advance_prob` (spec §3.5).

Everything downstream flows from the one matrix ``P`` (spec §0). We do not modify
the ratings / model / knockout modules; we only call them.

Leak-free note: these are *current-strength* forecasts for future 2026 fixtures,
so we use ``teams.elo_current`` (each team's latest rating). For a historical
walk-forward backtest, eval-engineer instead pairs ``rating_as_of`` with
``fit_up_to`` — that path is out of scope here (spec §3.6).

Public API
----------
``load_coeffs``       read the fitted ``(c0, c1, rho)`` for a model version.
``predict_fixture``   full pipeline for one ``fixture_id`` -> structured result.
``main``              ``python -m src.predict.fixture`` -> populate ``predictions``.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.model.dixon_coles import score_matrix
from src.model.knockout import advance_prob
from src.model.lambdas import predict_lambdas
from src.predict.markets import derive_markets
from src.ratings.elo import _home_bonus

# Stages (in the seeded 2026 fixtures) that are knockout ties requiring a winner:
# every remaining World Cup match is knockout, but we key off the stage string so
# a future league fixture (a plain 90' match) would simply skip the KO branch.
_KO_STAGES = {
    "round-of-16",
    "round of 16",
    "quarter-final",
    "quarterfinal",
    "semi-final",
    "semifinal",
    "third place",
    "final",
}


def is_knockout_stage(stage: str | None) -> bool:
    """True if ``stage`` is a knockout tie (needs ET + penalties, spec §3.5)."""
    if stage is None:
        return False
    return stage.strip().lower() in _KO_STAGES


def load_coeffs(con, model_version: str = "dc-v1") -> tuple[float, float, float]:
    """Read the fitted ``(c0, c1, rho)`` for ``model_version`` from ``model_coeffs``.

    The goals model persists its full-history fit there (see
    :func:`src.model.dixon_coles.persist_coeffs`). We take the full-history row
    (``fit_up_to IS NULL``) — the production coefficients for forecasting future
    fixtures — falling back to the most recent row for the version if, for some
    reason, no full-history row exists.
    """
    row = con.execute(
        """
        SELECT c0, c1, rho FROM model_coeffs
        WHERE model_version = ? AND fit_up_to IS NULL
        ORDER BY created_at DESC LIMIT 1
        """,
        [model_version],
    ).fetchone()
    if row is None:
        row = con.execute(
            "SELECT c0, c1, rho FROM model_coeffs WHERE model_version = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [model_version],
        ).fetchone()
    if row is None:
        raise ValueError(
            f"no fitted coefficients for model_version={model_version!r}; "
            "run `python -m src.model.dixon_coles` first"
        )
    return float(row[0]), float(row[1]), float(row[2])


def _team_elo(con, team_id: int) -> float:
    """Current strength ``teams.elo_current`` for ``team_id`` (base if missing)."""
    row = con.execute(
        "SELECT elo_current FROM teams WHERE team_id = ?", [int(team_id)]
    ).fetchone()
    if row is None or row[0] is None:
        return float(load_config().base_rating)
    return float(row[0])


def predict_fixture(
    fixture_id: int,
    con=None,
    model_version: str = "dc-v1",
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Run the full prediction pipeline for one fixture (spec §1, §3.4, §3.5).

    Looks up the fixture, both teams' current Elo (honouring ``neutral`` /
    ``host_flag`` for ``H_eff``), the fitted coefficients, then::

        (lam, mu) = predict_lambdas(elo_home, elo_away, H_eff, c0, c1)
        P         = score_matrix(lam, mu, rho, K)
        markets   = derive_markets(P)            # §3.4, all summed from P
        advance   = advance_prob(...)            # §3.5, only if knockout stage

    Returns a structured dict. The keys ``p_home``/``p_draw``/``p_away``,
    ``most_likely_score`` and ``matrix_json`` (JSON-encoded ``P``) are exactly the
    columns of the ``predictions`` table (spec §2.2), so the result serialises
    straight to the DB (see :func:`_row_for_predictions`).

    Supply a DuckDB ``con`` or one is opened read-only from config.
    """
    if cfg is None:
        cfg = load_config()

    own_con = False
    if con is None:
        import duckdb

        con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
        own_con = True

    try:
        fx = con.execute(
            "SELECT fixture_id, date, home_id, away_id, stage, neutral, host_flag "
            "FROM fixtures WHERE fixture_id = ?",
            [int(fixture_id)],
        ).fetchone()
        if fx is None:
            raise ValueError(f"no fixture with fixture_id={fixture_id}")
        _, date, home_id, away_id, stage, neutral, host_flag = fx
        home_id, away_id = int(home_id), int(away_id)

        c0, c1, rho = load_coeffs(con, model_version)
        elo_home = _team_elo(con, home_id)
        elo_away = _team_elo(con, away_id)

        # Effective home bonus, identical rule to the Elo engine (spec §3.1/§7.3).
        h_eff = _home_bonus(bool(neutral), bool(host_flag), cfg.H)

        lam, mu = predict_lambdas(elo_home, elo_away, h_eff, c0, c1)
        P = score_matrix(lam, mu, rho, K=cfg.K_max)

        markets = derive_markets(P)

        result: dict[str, Any] = {
            "fixture_id": int(fixture_id),
            "model_version": model_version,
            "date": date,
            "home_id": home_id,
            "away_id": away_id,
            "stage": stage,
            "neutral": bool(neutral),
            "host_flag": bool(host_flag),
            "h_eff": h_eff,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "lambda_home": lam,
            "lambda_away": mu,
            "rho": rho,
            "matrix": P,
            **markets,
        }

        # Knockout ties additionally carry qualification probabilities (spec §3.5).
        if is_knockout_stage(stage):
            adv = advance_prob(
                lam, mu, rho, elo_home, elo_away,
                theta=cfg.theta, kappa=cfg.kappa, K=cfg.K_max,
            )
            result["advance"] = adv

        return result
    finally:
        if own_con:
            con.close()


def _row_for_predictions(result: dict[str, Any]) -> list[Any]:
    """Flatten a :func:`predict_fixture` result into a ``predictions`` row.

    Columns (spec §2.2): ``fixture_id, model_version, matrix_json, p_home,
    p_draw, p_away, most_likely_score, created_at``. ``matrix_json`` is the full
    ``P`` as a JSON list-of-lists so it round-trips losslessly to the API.
    """
    x, y = result["most_likely_score"]
    matrix_json = json.dumps(np.asarray(result["matrix"]).tolist())
    return [
        result["fixture_id"],
        result["model_version"],
        matrix_json,
        result["p_home"],
        result["p_draw"],
        result["p_away"],
        f"{x}-{y}",
    ]


def write_predictions(con, results: list[dict[str, Any]]) -> int:
    """Upsert prediction rows into the ``predictions`` table; return rows written.

    Idempotent per ``(fixture_id, model_version)``: existing rows for the same
    pair are deleted first, so re-running ``main`` refreshes rather than
    duplicates.
    """
    for result in results:
        con.execute(
            "DELETE FROM predictions WHERE fixture_id = ? AND model_version = ?",
            [result["fixture_id"], result["model_version"]],
        )
        row = _row_for_predictions(result)
        con.execute(
            "INSERT INTO predictions "
            "(fixture_id, model_version, matrix_json, p_home, p_draw, p_away, "
            "most_likely_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, now())",
            row,
        )
    return len(results)


def matrix_from_json(matrix_json: str) -> np.ndarray:
    """Inverse of the ``matrix_json`` encoding: JSON list-of-lists -> ``np.ndarray``."""
    return np.asarray(json.loads(matrix_json), dtype=float)


def main(argv=None) -> None:
    """Predict every row of ``fixtures`` and write them to ``predictions``.

    ``python -m src.predict.fixture``. Reads the fitted ``dc-v1`` coefficients,
    runs :func:`predict_fixture` for each fixture, prints a one-line summary per
    fixture (with advance probs for knockout ties) and persists the rows.
    """
    import duckdb

    cfg = load_config()
    model_version = "dc-v1"

    con = duckdb.connect(str(cfg.paths.db_path))
    try:
        fixture_ids = [
            int(r[0])
            for r in con.execute(
                "SELECT fixture_id FROM fixtures ORDER BY date, fixture_id"
            ).fetchall()
        ]
        # Team names for a readable printout.
        names = dict(
            con.execute("SELECT team_id, name_canonical FROM teams").fetchall()
        )

        results = [
            predict_fixture(fid, con=con, model_version=model_version, cfg=cfg)
            for fid in fixture_ids
        ]

        print(f"[predict] {len(results)} fixtures predicted (model={model_version}):\n")
        for r in results:
            hn = names.get(r["home_id"], r["home_id"])
            an = names.get(r["away_id"], r["away_id"])
            line = (
                f"  #{r['fixture_id']} {r['stage']:<14} "
                f"{hn} vs {an}: "
                f"ML {r['most_likely_score_str']}  "
                f"1N2 {r['p_home']:.2f}/{r['p_draw']:.2f}/{r['p_away']:.2f}  "
                f"O2.5 {r['over_under']['2.5']['over']:.2f}  "
                f"BTTS {r['btts']:.2f}"
            )
            if "advance" in r:
                line += (
                    f"  qualifies {r['advance']['p_home_advance']:.2f}/"
                    f"{r['advance']['p_away_advance']:.2f}"
                )
            print(line)

        n = write_predictions(con, results)
        con.commit()
        print(f"\n[predict] {n} rows written to predictions (matrix_json included).")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    main()
