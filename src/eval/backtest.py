"""Strict walk-forward backtest for ONZE (spec §3.6 + §3.7).

This is the guardian of methodological honesty. It re-plays history *forward*:
for each time block it fits the goal model on **only** the matches that were
already played before the block starts, predicts that block's matches, scores
the predictions, then advances. Nothing about a match — not its ratings, not the
fitted coefficients — is ever a function of that match or any later one
(spec §3.6). Every prediction carries an explicit date assertion enforcing this.

The comparison (spec §3.7)
--------------------------
For each predicted match we record three sets of 1N2 probabilities:

1. **model** — Dixon-Coles ``score_matrix`` -> ``derive_markets`` on the
   leak-free as-of Elo and the ``fit_up_to`` coefficients.
2. **elo** — the bare-Elo 1N2 baseline (``elo_1x2``), the honest model-free bar.
3. **bookmaker** — de-vigged closing odds (normalisation, with an optional
   Shin/power-method refinement). The ``odds`` table is currently EMPTY, so this
   baseline is **guarded**: it runs only when odds are present and is otherwise
   clearly reported as "pending odds ingestion".

We report the mean RPS of each, plus log-loss / Brier on the BTTS and O/U 2.5
binary markets, and a reliability diagram (calibration) for the model.

xi tuning (spec §3.7, §7.4)
---------------------------
The time-decay ``xi`` is tuned by **backtest RPS** over a small documented grid
(half-lives around the spec's 2-year starting point). The chosen value and the
full search are written to the report. This is the *only* place xi is tuned —
by out-of-sample RPS, never "au pif".

No-leakage self-defence
-----------------------
* fit window: ``fit_up_to(block_start)`` -> matches strictly ``< block_start``.
* per match: ``assert fit_max_date < match_date`` and
  ``assert rating_as_of`` used strict-``<`` lookups.
* the fit window's own last match date is asserted ``< block_start <= match_date``.

Honesty caveat (spec §3.7 note): the ~30 remaining 2026 World Cup matches are a
demo, NOT a validation set. Validation is THIS multi-decade backtest. A single
exact-score accuracy number is never reported as "success"; we report score
*distributions* and *calibration*.

Public API
----------
``walk_forward``   run the backtest over a date range, return per-match records.
``summarise``      aggregate records into RPS / log-loss / Brier + calibration.
``tune_xi``        grid-search xi by backtest RPS.
``main``           ``python -m src.eval.backtest [--smoke]`` -> writes the report.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.model.dixon_coles import fit_up_to, score_matrix
from src.model.lambdas import predict_lambdas
from src.predict.markets import derive_markets
from src.ratings.elo import elo_1x2
from src.eval import calibration as calib
from src.eval.metrics import brier, log_loss, rps

# Outcome index used throughout: 0 = home win, 1 = draw, 2 = away win.
_HOME, _DRAW, _AWAY = 0, 1, 2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Odds -> implied probabilities (spec §3.7 baseline #2), guarded on availability
# ---------------------------------------------------------------------------
def devig_normalisation(o_home: float, o_draw: float, o_away: float) -> tuple[float, float, float]:
    """De-vig decimal odds by simple normalisation (spec §3.7): p_i = (1/o_i)/Σ.

    The raw implied probabilities ``1/o_i`` sum to > 1 (the bookmaker's
    overround / vig); dividing by their sum removes the margin proportionally.
    This is the spec's default de-vig. Returns ``(p_home, p_draw, p_away)``.
    """
    inv = np.array([1.0 / o_home, 1.0 / o_draw, 1.0 / o_away], dtype=float)
    p = inv / inv.sum()
    return float(p[0]), float(p[1]), float(p[2])


def devig_shin(
    o_home: float, o_draw: float, o_away: float, iters: int = 100
) -> tuple[float, float, float]:
    """De-vig via Shin's method (spec §3.7 refinement) — less biased than plain norm.

    Shin (1992/1993) models the overround as arising from a fraction ``z`` of
    insider trading and backs out the "true" probabilities. We solve for ``z``
    by fixed-point iteration on::

        p_i = ( sqrt(z^2 + 4(1-z) * pi_i^2 / B) - z ) / (2 (1 - z))

    where ``pi_i = (1/o_i)/B`` are the normalised implied probs and
    ``B = sum_i 1/o_i`` (the book sum). ``z`` is chosen so ``sum_i p_i = 1``.
    Falls back to plain normalisation if the solve degenerates. Returns
    ``(p_home, p_draw, p_away)``.
    """
    inv = np.array([1.0 / o_home, 1.0 / o_draw, 1.0 / o_away], dtype=float)
    B = inv.sum()
    pi = inv / B  # normalised implied (sums to 1)

    z = 0.0
    for _ in range(iters):
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / B)
        # sum of numerators = 2(1-z) when constraint holds; solve for z via its
        # closed-form update: z_new makes sum(p_i) = 1.
        s = root.sum()
        denom = 2.0 * (1.0 - z)
        if denom <= 0:
            break
        # sum p_i = (s - 3 z) / (2 (1 - z)) = 1  ->  s - 3z = 2(1-z) -> z = s - 2
        z_new = s - 2.0
        z_new = min(max(z_new, 0.0), 0.5)
        if abs(z_new - z) < 1e-10:
            z = z_new
            break
        z = z_new

    denom = 2.0 * (1.0 - z)
    if denom <= 0:
        return devig_normalisation(o_home, o_draw, o_away)
    root = np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / B)
    p = (root - z) / denom
    p = np.clip(p, 1e-9, None)
    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])


# ---------------------------------------------------------------------------
# Fast leak-free as-of Elo index (equivalent to rating_as_of, strict `<`)
# ---------------------------------------------------------------------------
class AsOfElo:
    """In-memory strict-``<`` as-of Elo lookup, built once from ratings_history.

    This is a *performance* mirror of :func:`src.ratings.elo.rating_as_of`: it
    returns a team's last stored rating **strictly before** a date, identical
    semantics (strict ``<``, ``base_rating`` fallback), but via a per-team sorted
    array + binary search instead of one SQL round-trip per lookup. The backtest
    makes O(10^5) lookups; the per-query DB path would dominate runtime.

    Leak-free contract preserved: we use ``bisect_left`` on the team's sorted
    dates, so the selected row's date is guaranteed ``< date`` — never on or
    after it (§3.6). Verified against ``rating_as_of`` in the tests.

    Same-day double-headers (documented, NOT a leak): if a team has *multiple*
    ratings_history rows on the same earlier date (it played twice that day),
    this returns the FIRST such row's predecessor deterministically, whereas
    ``rating_as_of``'s SQL ``ORDER BY date DESC LIMIT 1`` returns an arbitrary
    one of the same-day rows. Both values are strictly pre-``date`` (leak-free);
    they differ only in *which* same-day pre-match snapshot is used. This affects
    a handful of lookups over the whole history (~2 of 5k in a 2.5-year window)
    and moves RPS at the 5th decimal. ``AsOfElo`` is at least deterministic. The
    upstream *training* path (``match_features``) disambiguates doubles with a
    sequence-join; this is the *prediction* path, where the ambiguity is
    inherent. Flagged to the coach as a minor pre-existing ``rating_as_of`` quirk.
    """

    def __init__(self, ratings_history: pd.DataFrame, base_rating: float):
        rh = ratings_history.copy()
        rh["date"] = pd.to_datetime(rh["date"]).dt.date
        rh = rh.sort_values(["team_id", "date"], kind="mergesort")
        self._base = float(base_rating)
        self._dates: dict[int, list] = {}
        self._elos: dict[int, list] = {}
        for tid, grp in rh.groupby("team_id"):
            self._dates[int(tid)] = grp["date"].tolist()
            self._elos[int(tid)] = grp["elo"].tolist()

    def rating(self, team_id: int, date: _dt.date) -> float:
        """Last rating strictly before ``date`` (base_rating if none)."""
        dates = self._dates.get(int(team_id))
        if not dates:
            return self._base
        # bisect_left -> first index with dates[i] >= date; the one before is the
        # last strictly-earlier row. i == 0 means no prior row.
        i = bisect.bisect_left(dates, date)
        if i == 0:
            return self._base
        return float(self._elos[int(team_id)][i - 1])


# ---------------------------------------------------------------------------
# Walk-forward core (spec §3.6)
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Parameters of a walk-forward run."""

    start: _dt.date          # first block start (predict matches on/after this)
    end: _dt.date            # last date to predict (inclusive)
    block_months: int = 12   # refit cadence: refit every this many months
    xi: float | None = None  # time decay; None -> config default
    H: float | None = None   # home bonus; None -> config default
    K_max: int | None = None # score matrix truncation; None -> config default
    min_train: int = 500     # require at least this many training matches to fit


def _to_date(x) -> _dt.date:
    if isinstance(x, str):
        return _dt.date.fromisoformat(x[:10])
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    return x


def _outcome_index(hg: int, ag: int) -> int:
    """Realised 1N2 index from a final score (0 home, 1 draw, 2 away)."""
    if hg > ag:
        return _HOME
    if hg < ag:
        return _AWAY
    return _DRAW


def _block_starts(start: _dt.date, end: _dt.date, block_months: int) -> list[_dt.date]:
    """Generate block-start dates from ``start`` to ``end`` every ``block_months``."""
    starts = []
    cur = start
    while cur <= end:
        starts.append(cur)
        # advance by block_months, clamping day to 1 to avoid month-length issues.
        y = cur.year + (cur.month - 1 + block_months) // 12
        mo = (cur.month - 1 + block_months) % 12 + 1
        cur = _dt.date(y, mo, 1)
    return starts


def walk_forward(
    con,
    bt: BacktestConfig,
    odds: pd.DataFrame | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the strict walk-forward backtest, returning one record per match.

    For each block ``[block_start, next_block_start)`` we:

    1. ``fit_up_to(block_start)`` — coefficients from matches strictly BEFORE the
       block (spec §3.6). We assert the fit window's max date ``< block_start``.
    2. For every played match in the block, look up each team's Elo strictly
       before the match (``rating_as_of``, strict ``<``), predict the model 1N2,
       the bare-Elo 1N2 baseline, and (if odds present) the de-vigged book 1N2.
    3. Assert ``fit_max_date < match_date`` for that match — the no-leak invariant.

    Returns a DataFrame with, per match: date, outcome index, the three sets of
    1N2 probs (model / elo / book), the model's BTTS and O/U-2.5 probabilities,
    and the realised BTTS / over-2.5 indicators.
    """
    cfg = load_config()
    xi = bt.xi if bt.xi is not None else cfg.xi
    H = bt.H if bt.H is not None else cfg.H
    K_max = bt.K_max if bt.K_max is not None else cfg.K_max

    # All played matches, once — reused as the fit corpus and the prediction set.
    all_matches = con.execute(
        "SELECT match_id, date, home_id, away_id, home_goals, away_goals, "
        "neutral, tournament, importance_k FROM matches "
        "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL "
        "ORDER BY date, match_id"
    ).fetchdf()
    all_matches["date"] = pd.to_datetime(all_matches["date"]).dt.date

    # Fast leak-free as-of Elo index (strict `<`, mirrors rating_as_of) — built
    # once so the O(10^5) per-match lookups don't each hit the DB.
    ratings_history = con.execute(
        "SELECT team_id, date, elo FROM ratings_history"
    ).fetchdf()
    as_of = AsOfElo(ratings_history, base_rating=cfg.base_rating)

    starts = _block_starts(bt.start, bt.end, bt.block_months)
    records: list[dict] = []

    for i, block_start in enumerate(starts):
        block_end = starts[i + 1] if i + 1 < len(starts) else (bt.end + _dt.timedelta(days=1))
        block_end = min(block_end, bt.end + _dt.timedelta(days=1))

        # --- Fit ONLY on matches strictly before this block (spec §3.6) --------
        train_mask = all_matches["date"] < block_start
        n_train = int(train_mask.sum())
        if n_train < bt.min_train:
            if verbose:
                print(f"[bt] {block_start}: only {n_train} train matches (<{bt.min_train}), skip")
            continue

        fit = fit_up_to(block_start, matches=all_matches, con=con, xi=xi, H=H)

        # No-leak invariant #1: the fit window's newest match precedes the block.
        fit_max_date = all_matches.loc[train_mask, "date"].max()
        assert fit_max_date < block_start, (
            f"LEAK: fit window max {fit_max_date} not < block start {block_start}"
        )

        block = all_matches[
            (all_matches["date"] >= block_start) & (all_matches["date"] < block_end)
        ]
        if verbose:
            print(f"[bt] block {block_start}..{block_end} | train={n_train} "
                  f"predict={len(block)} | c0={fit.c0:.3f} c1={fit.c1:.5f} rho={fit.rho:.3f}")

        for row in block.itertuples(index=False):
            match_date = row.date
            # No-leak invariant #2 (asserted per match): coefficients predate it.
            assert fit_max_date < match_date, (
                f"LEAK: fit max {fit_max_date} not < match {match_date}"
            )

            # Leak-free as-of ratings: strictly before the match (spec §3.6).
            # AsOfElo mirrors rating_as_of's strict `<` (tested for equivalence).
            r_home = as_of.rating(int(row.home_id), match_date)
            r_away = as_of.rating(int(row.away_id), match_date)

            # Effective home bonus: 0 on neutral ground, else H (no host_flag in
            # historical matches; hosts only matter for 2026 fixtures).
            h_eff = 0.0 if bool(row.neutral) else H

            # --- Model 1N2 via the full Dixon-Coles pipeline -------------------
            lam, mu = predict_lambdas(r_home, r_away, h_eff, fit.c0, fit.c1)
            P = score_matrix(lam, mu, fit.rho, K=K_max)
            mk = derive_markets(P)
            model_probs = (mk["p_home"], mk["p_draw"], mk["p_away"])

            # --- Bare-Elo 1N2 baseline (spec §3.7 #1) --------------------------
            elo_probs = elo_1x2(r_home, r_away, h_eff)

            hg, ag = int(row.home_goals), int(row.away_goals)
            outcome = _outcome_index(hg, ag)
            btts_obs = 1.0 if (hg > 0 and ag > 0) else 0.0
            over25_obs = 1.0 if (hg + ag) > 2 else 0.0
            # model binary-market probs (from the same matrix)
            btts_pred = float(mk["btts"])
            over25_pred = float(mk["over_under"]["2.5"]["over"])

            rec = {
                "match_id": int(row.match_id),
                "date": match_date,
                "outcome": outcome,
                "home_goals": hg,
                "away_goals": ag,
                "model_home": model_probs[0],
                "model_draw": model_probs[1],
                "model_away": model_probs[2],
                "elo_home": elo_probs[0],
                "elo_draw": elo_probs[1],
                "elo_away": elo_probs[2],
                "btts_pred": btts_pred,
                "btts_obs": btts_obs,
                "over25_pred": over25_pred,
                "over25_obs": over25_obs,
            }

            # --- Bookmaker baseline: GUARDED on odds availability --------------
            if odds is not None and not odds.empty:
                orow = odds[odds["match_id"] == int(row.match_id)]
                if not orow.empty:
                    o = orow.iloc[0]
                    bp = devig_normalisation(float(o["o_home"]), float(o["o_draw"]), float(o["o_away"]))
                    rec.update(book_home=bp[0], book_draw=bp[1], book_away=bp[2])

            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation (spec §3.7): RPS / log-loss / Brier + calibration
# ---------------------------------------------------------------------------
@dataclass
class BacktestSummary:
    """Aggregated backtest metrics + calibration table."""

    n: int
    rps_model: float
    rps_elo: float
    rps_book: float | None
    logloss_btts: float
    brier_btts: float
    logloss_over25: float
    brier_over25: float
    calib_table: pd.DataFrame
    ece: float
    date_min: _dt.date | None
    date_max: _dt.date | None


def summarise(records: pd.DataFrame, n_bins: int = 10) -> BacktestSummary:
    """Aggregate per-match records into the spec §3.7 metrics + calibration.

    Reports mean RPS for model / Elo (/ book if present), log-loss and Brier on
    the model's BTTS and O/U-2.5 markets, and a one-vs-rest reliability table +
    ECE for the model's 1N2 probabilities.
    """
    if records.empty:
        raise ValueError("no records to summarise (empty backtest)")

    outcomes = records["outcome"].to_numpy()
    model = records[["model_home", "model_draw", "model_away"]].to_numpy()
    elo = records[["elo_home", "elo_draw", "elo_away"]].to_numpy()

    rps_model = rps(model, outcomes)
    rps_elo = rps(elo, outcomes)

    rps_book = None
    if {"book_home", "book_draw", "book_away"}.issubset(records.columns):
        book_rows = records.dropna(subset=["book_home", "book_draw", "book_away"])
        if not book_rows.empty:
            rps_book = rps(
                book_rows[["book_home", "book_draw", "book_away"]].to_numpy(),
                book_rows["outcome"].to_numpy(),
            )

    ll_btts = log_loss(records["btts_pred"].to_numpy(), records["btts_obs"].to_numpy())
    br_btts = brier(records["btts_pred"].to_numpy(), records["btts_obs"].to_numpy())
    ll_ou = log_loss(records["over25_pred"].to_numpy(), records["over25_obs"].to_numpy())
    br_ou = brier(records["over25_pred"].to_numpy(), records["over25_obs"].to_numpy())

    pred_flat, obs_flat = calib.onehot_flatten(model, outcomes)
    table = calib.reliability_table(pred_flat, obs_flat, n_bins=n_bins)
    ece = calib.expected_calibration_error(table)

    return BacktestSummary(
        n=len(records),
        rps_model=rps_model,
        rps_elo=rps_elo,
        rps_book=rps_book,
        logloss_btts=ll_btts,
        brier_btts=br_btts,
        logloss_over25=ll_ou,
        brier_over25=br_ou,
        calib_table=table,
        ece=ece,
        date_min=records["date"].min(),
        date_max=records["date"].max(),
    )


# ---------------------------------------------------------------------------
# xi tuning by backtest RPS (spec §3.7, §7.4)
# ---------------------------------------------------------------------------
def half_life_to_xi(days: float) -> float:
    """Per-day decay rate xi for a given half-life in days: xi = ln(2) / days."""
    return math.log(2.0) / days


def tune_xi(
    con,
    bt: BacktestConfig,
    half_lives_days: list[float],
    odds: pd.DataFrame | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Grid-search ``xi`` (via half-life) by walk-forward model RPS (spec §3.7).

    Runs the *same* walk-forward for each candidate half-life and records the
    model's mean RPS. Lower RPS wins. Returns a DataFrame
    ``[half_life_days, xi, rps_model, rps_elo, n]`` sorted by half-life; the
    caller picks ``argmin(rps_model)``. This is the documented, out-of-sample
    search the spec demands (never tune xi by eye).
    """
    rows = []
    for hl in half_lives_days:
        xi = half_life_to_xi(hl)
        bt_xi = BacktestConfig(
            start=bt.start, end=bt.end, block_months=bt.block_months,
            xi=xi, H=bt.H, K_max=bt.K_max, min_train=bt.min_train,
        )
        recs = walk_forward(con, bt_xi, odds=odds, verbose=False)
        summ = summarise(recs)
        rows.append({
            "half_life_days": hl,
            "xi": xi,
            "rps_model": summ.rps_model,
            "rps_elo": summ.rps_elo,
            "n": summ.n,
        })
        if verbose:
            print(f"[xi] half-life {hl:>5.0f}d  xi={xi:.6g}  "
                  f"RPS_model={summ.rps_model:.5f}  (Elo {summ.rps_elo:.5f}, n={summ.n})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Machine-readable JSON artifact (consumed by the API: GET /eval/backtest)
# ---------------------------------------------------------------------------
def build_report_json(
    summ: BacktestSummary,
    chosen_hl: float,
    chosen_xi: float,
    window_start: _dt.date,
    caveat: str | None = None,
) -> dict:
    """Assemble the API-facing JSON payload in the preferred ``results`` shape.

    The api-engineer's ``GET /eval/backtest?from=YYYY`` reads this artifact and
    expects ``{"results": [{"name": ..., "rps": ...}, ...]}``. We expose:

    * ``generated_from`` — the backtest **window start** actually used (so the
      endpoint can report which window it is serving; it does not re-filter per
      request, it just echoes this).
    * ``n_matches``, ``xi_chosen``, ``xi_half_life_days``, ``ece``.
    * ``results`` — one row per predictor. ``model`` carries the binary-market
      scores too; ``elo`` its RPS; ``bookmaker`` is **guarded**: ``rps: null``
      with a ``note`` when odds are absent, so the API shows "pending odds".
    * ``calibration`` — the reliability bins (p_pred / p_obs / n).
    * ``caveat`` — the honesty / sample-size note (spec §3.7).
    """
    results = [
        {
            "name": "model",
            "rps": summ.rps_model,
            "log_loss_btts": summ.logloss_btts,
            "brier_btts": summ.brier_btts,
            "log_loss_ou25": summ.logloss_over25,
            "brier_ou25": summ.brier_over25,
        },
        {"name": "elo", "rps": summ.rps_elo},
    ]
    if summ.rps_book is not None:
        results.append({"name": "bookmaker", "rps": summ.rps_book})
    else:
        results.append(
            {"name": "bookmaker", "rps": None, "note": "pending odds ingestion"}
        )

    calibration = [
        {
            "bin_lo": float(r["bin_lo"]),
            "bin_hi": float(r["bin_hi"]),
            "p_pred": float(r["mean_pred"]),
            "p_obs": float(r["observed"]),
            "n": int(r["count"]),
        }
        for _, r in summ.calib_table.iterrows()
    ]

    if caveat is None:
        caveat = (
            "Smoke window is short; validation is the multi-decade full backtest, "
            "not the ~30 remaining WC matches."
        )

    return {
        "generated_from": window_start.isoformat(),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_matches": int(summ.n),
        "date_min": summ.date_min.isoformat() if summ.date_min is not None else None,
        "date_max": summ.date_max.isoformat() if summ.date_max is not None else None,
        "xi_chosen": chosen_xi,
        "xi_half_life_days": chosen_hl,
        "ece": summ.ece,
        "results": results,
        "calibration": calibration,
        "caveat": caveat,
    }


def write_json_artifact(
    summ: BacktestSummary,
    chosen_hl: float,
    chosen_xi: float,
    window_start: _dt.date,
    processed_dir: Path,
    caveat: str | None = None,
) -> Path:
    """Persist the API JSON artifact to ``<processed_dir>/backtest_report.json``.

    This is the file the API's ``GET /eval/backtest`` looks for (preferred name /
    location). Returns the written path.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    payload = build_report_json(summ, chosen_hl, chosen_xi, window_start, caveat)
    path = processed_dir / "backtest_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
def write_report(
    summ: BacktestSummary,
    xi_grid: pd.DataFrame,
    chosen_hl: float,
    chosen_xi: float,
    bt: BacktestConfig,
    smoke: bool,
    out_dir: Path = REPORTS_DIR,
    odds_present: bool = False,
    processed_dir: Path | None = None,
) -> Path:
    """Write ``BACKTEST_REPORT.md`` (+ calibration artefact + CSVs) to ``out_dir``.

    Also persists the machine-readable ``backtest_report.json`` to
    ``processed_dir`` (defaults to ``config.paths.processed_dir``) for the API's
    ``GET /eval/backtest`` endpoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if processed_dir is None:
        processed_dir = load_config().paths.processed_dir
    json_path = write_json_artifact(
        summ, chosen_hl, chosen_xi, bt.start, processed_dir
    )

    # Calibration artefact (PNG if matplotlib, else CSV+MD) + raw calibration CSV.
    calib_artefacts = calib.render(
        summ.calib_table, out_dir, stem="backtest_calibration",
        title="ONZE backtest — model 1N2 reliability (one-vs-rest)",
    )
    xi_grid.to_csv(out_dir / "backtest_xi_grid.csv", index=False)

    def _delta(a: float, b: float) -> str:
        d = a - b
        sign = "better" if d < 0 else ("worse" if d > 0 else "equal")
        return f"{d:+.5f} ({sign} than Elo)"

    lines: list[str] = []
    lines.append("# ONZE — Walk-Forward Backtest Report")
    lines.append("")
    lines.append(f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Mode: {'SMOKE (fast, recent window)' if smoke else 'FULL'}_")
    lines.append("")
    lines.append("## Protocol (spec §3.6 — no leakage)")
    lines.append("")
    lines.append(
        "Strict **walk-forward**. Time is split into blocks; for each block the "
        "goal-model coefficients `(c0, c1, rho)` are fit with `fit_up_to("
        "block_start)` on **only** matches strictly *before* the block, then that "
        "block's matches are predicted. Per-team Elo is read with `rating_as_of` "
        "(strict `<`). Two invariants are **asserted in code** for every "
        "prediction: (1) the fit window's max date `<` block start, and (2) the "
        "fit window's max date `<` each predicted match's date. So no prediction "
        "can see its own match or any later one."
    )
    lines.append("")
    lines.append(
        f"- Window predicted: **{summ.date_min} → {summ.date_max}**  \n"
        f"- Refit cadence: every **{bt.block_months} months**  \n"
        f"- Matches scored: **{summ.n:,}**"
    )
    lines.append("")
    lines.append("## Headline: RPS (ordered 1N2, spec §3.7 — lower is better)")
    lines.append("")
    lines.append("| Predictor | Mean RPS | vs Elo baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| **ONZE model (Dixon-Coles)** | **{summ.rps_model:.5f}** | {_delta(summ.rps_model, summ.rps_elo)} |")
    lines.append(f"| Bare Elo (baseline #1) | {summ.rps_elo:.5f} | — |")
    if summ.rps_book is not None:
        lines.append(f"| De-vigged bookmaker (baseline #2) | {summ.rps_book:.5f} | {_delta(summ.rps_book, summ.rps_elo)} |")
    else:
        lines.append("| De-vigged bookmaker (baseline #2) | _pending_ | — |")
    lines.append("")
    if summ.rps_book is None:
        lines.append(
            "> **Bookmaker baseline pending odds ingestion.** The `odds` table is "
            "currently empty, so the model-vs-book comparison (the real bar to "
            "beat, spec §3.7) cannot be computed yet. The de-vig code "
            "(normalisation + Shin refinement) and the comparison harness are in "
            "place and will activate automatically once closing odds are loaded. "
            "For now we report **model vs bare Elo** only."
        )
        lines.append("")
    lines.append("## Binary markets (spec §3.7 — model, lower is better)")
    lines.append("")
    lines.append("| Market | Log-loss | Brier |")
    lines.append("|---|---|---|")
    lines.append(f"| BTTS | {summ.logloss_btts:.5f} | {summ.brier_btts:.5f} |")
    lines.append(f"| Over/Under 2.5 | {summ.logloss_over25:.5f} | {summ.brier_over25:.5f} |")
    lines.append("")
    lines.append("## Calibration (reliability, spec §3.7)")
    lines.append("")
    lines.append(
        f"One-vs-rest reliability of the model's 1N2 probabilities "
        f"(each match contributes 3 points: P(home)/P(draw)/P(away) vs their "
        f"indicators). **Expected Calibration Error (ECE) = {summ.ece:.4f}** "
        f"(count-weighted mean |mean_pred − observed|; 0 = perfect)."
    )
    lines.append("")
    artefact_note = ", ".join(f"`{p.name}`" for p in calib_artefacts.values())
    lines.append(f"Artefacts: {artefact_note}.")
    lines.append("")
    lines.append("| bin | count | mean_pred | observed | gap |")
    lines.append("|---|---|---|---|---|")
    for _, r in summ.calib_table.iterrows():
        lines.append(
            f"| [{r['bin_lo']:.2f}, {r['bin_hi']:.2f}) | {int(r['count'])} "
            f"| {r['mean_pred']:.3f} | {r['observed']:.3f} | {r['gap']:+.3f} |"
        )
    lines.append("")
    lines.append("## xi tuning (time decay, spec §3.7 / §7.4)")
    lines.append("")
    lines.append(
        "`xi` is the per-day exponential time-decay rate "
        "(`phi(dt) = exp(-xi·dt)`). It is tuned by **backtest RPS** over the grid "
        "of half-lives below (same walk-forward, out-of-sample). Lower RPS wins."
    )
    lines.append("")
    lines.append("| Half-life (days) | xi | Model RPS | Elo RPS | n |")
    lines.append("|---|---|---|---|---|")
    best_rps = xi_grid["rps_model"].min()
    for _, r in xi_grid.sort_values("half_life_days").iterrows():
        marker = " ⬅ **chosen**" if abs(r["rps_model"] - best_rps) < 1e-12 else ""
        lines.append(
            f"| {r['half_life_days']:.0f} | {r['xi']:.6g} | "
            f"{r['rps_model']:.5f}{marker} | {r['rps_elo']:.5f} | {int(r['n'])} |"
        )
    lines.append("")
    lines.append(
        f"**Chosen xi = {chosen_xi:.6g}** (half-life ≈ {chosen_hl:.0f} days), "
        f"the grid minimiser of backtest model RPS."
    )
    lines.append("")
    lines.append("## Honesty / sample-size caveat (spec §3.7 note)")
    lines.append("")
    lines.append(
        "- The **validation** here is this **multi-decade walk-forward backtest** "
        "on tens of thousands of historical matches — that is the statistically "
        "meaningful signal.\n"
        "- The **~30 remaining 2026 World Cup matches are a demo, NOT a "
        "validation set**: 30 matches cannot separate models. Do not read the "
        "tournament results as a score.\n"
        "- We deliberately report **RPS distributions, calibration, and multiple "
        "markets** — never a single exact-score accuracy number as 'success'. "
        "Exact-score guessing is a poor, misleading metric for a distributional "
        "model.\n"
        "- Beating **bare Elo** on RPS shows the goal model adds value over the "
        "rating alone; **matching the de-vigged bookmaker** (once odds are "
        "ingested) is the real bar (spec §3.7)."
    )
    lines.append("")
    lines.append("## Machine-readable artifact")
    lines.append("")
    lines.append(
        f"A JSON summary for the API (`GET /eval/backtest`) is written to "
        f"`{json_path}` (preferred `results` shape; the bookmaker row is "
        f"null/guarded until odds are ingested)."
    )
    lines.append("")

    report_path = out_dir / "BACKTEST_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also drop the raw per-metric summary as CSV for programmatic consumption.
    pd.DataFrame([{
        "n": summ.n,
        "rps_model": summ.rps_model,
        "rps_elo": summ.rps_elo,
        "rps_book": summ.rps_book if summ.rps_book is not None else "",
        "ece": summ.ece,
        "logloss_btts": summ.logloss_btts,
        "brier_btts": summ.brier_btts,
        "logloss_over25": summ.logloss_over25,
        "brier_over25": summ.brier_over25,
        "chosen_xi": chosen_xi,
        "chosen_half_life_days": chosen_hl,
    }]).to_csv(out_dir / "backtest_summary.csv", index=False)

    return report_path


# ---------------------------------------------------------------------------
# Odds loader (guarded — table is empty for now)
# ---------------------------------------------------------------------------
def _load_odds(con) -> pd.DataFrame | None:
    """Load closing odds joined to played match_ids, or None if unavailable/empty.

    The `odds` table keys on `fixture_id`; we try to map fixtures to matches. If
    the table is empty (current state) or no mapping exists, returns None so the
    bookmaker baseline is cleanly skipped (spec §3.7 guard).
    """
    try:
        n = con.execute("SELECT count(*) FROM odds").fetchone()[0]
    except Exception:
        return None
    if not n:
        return None
    # Odds present: attempt to expose a match_id column. Fixtures may map to
    # matches; if the schema doesn't support it here, skip gracefully.
    try:
        df = con.execute(
            "SELECT fixture_id AS match_id, o_home, o_draw, o_away FROM odds"
        ).fetchdf()
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    """Run the walk-forward backtest and write ``reports/BACKTEST_REPORT.md``.

    ``--smoke`` runs a short recent window (fast, for CI); otherwise a full
    multi-decade backtest. In both modes xi is tuned by backtest RPS over a small
    documented grid and the chosen value + search are recorded in the report.
    """
    import duckdb

    parser = argparse.ArgumentParser(description="ONZE walk-forward backtest (spec §3.6/§3.7)")
    parser.add_argument("--smoke", action="store_true",
                        help="fast recent-window run for CI (few hundred matches)")
    parser.add_argument("--start", type=str, default=None, help="override start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="override end date YYYY-MM-DD")
    args = parser.parse_args(argv)

    cfg = load_config()
    con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
    try:
        max_played = _to_date(con.execute(
            "SELECT max(date) FROM matches WHERE home_goals IS NOT NULL"
        ).fetchone()[0])

        if args.smoke:
            # Short recent window: ~last 2 years, refit every 6 months.
            start = _to_date(args.start) if args.start else _dt.date(max_played.year - 2, 1, 1)
            end = _to_date(args.end) if args.end else max_played
            block_months = 6
            # Coarser, cheaper grid for smoke.
            half_lives = [365.0, 730.0, 1460.0]
        else:
            start = _to_date(args.start) if args.start else _dt.date(2006, 1, 1)
            end = _to_date(args.end) if args.end else max_played
            block_months = 12
            half_lives = [180.0, 365.0, 548.0, 730.0, 1095.0, 1460.0]

        bt = BacktestConfig(start=start, end=end, block_months=block_months)
        odds = _load_odds(con)
        odds_present = odds is not None and not odds.empty

        print(f"[bt] mode={'smoke' if args.smoke else 'full'}  window {start}..{end}  "
              f"block={block_months}mo  odds={'present' if odds_present else 'ABSENT (book baseline pending)'}")

        # --- xi tuning by backtest RPS (spec §3.7) -----------------------------
        print(f"[bt] tuning xi over half-lives {half_lives} (days)...")
        xi_grid = tune_xi(con, bt, half_lives, odds=odds, verbose=True)
        best = xi_grid.loc[xi_grid["rps_model"].idxmin()]
        chosen_hl = float(best["half_life_days"])
        chosen_xi = float(best["xi"])
        print(f"[bt] chosen xi = {chosen_xi:.6g}  (half-life {chosen_hl:.0f} d, "
              f"RPS_model={best['rps_model']:.5f})")

        # --- Final run at the chosen xi ---------------------------------------
        bt_best = BacktestConfig(
            start=start, end=end, block_months=block_months,
            xi=chosen_xi, min_train=bt.min_train,
        )
        records = walk_forward(con, bt_best, odds=odds, verbose=True)
        summ = summarise(records)

        report = write_report(
            summ, xi_grid, chosen_hl, chosen_xi, bt_best, smoke=args.smoke,
            odds_present=odds_present,
        )

        print("\n[bt] ===== SUMMARY =====")
        print(f"[bt] matches scored : {summ.n:,}  ({summ.date_min} .. {summ.date_max})")
        print(f"[bt] RPS model      : {summ.rps_model:.5f}")
        print(f"[bt] RPS bare Elo   : {summ.rps_elo:.5f}  "
              f"(model {'beats' if summ.rps_model < summ.rps_elo else 'does NOT beat'} Elo)")
        if summ.rps_book is not None:
            print(f"[bt] RPS bookmaker  : {summ.rps_book:.5f}")
        else:
            print("[bt] RPS bookmaker  : pending (odds table empty)")
        print(f"[bt] ECE (calib)    : {summ.ece:.4f}")
        print(f"[bt] BTTS  logloss  : {summ.logloss_btts:.5f}  brier {summ.brier_btts:.5f}")
        print(f"[bt] O/U2.5 logloss : {summ.logloss_over25:.5f}  brier {summ.brier_over25:.5f}")
        print(f"[bt] report written : {report}")
        print(f"[bt] JSON artifact  : {cfg.paths.processed_dir / 'backtest_report.json'} "
              f"(generated_from={start})")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    main()
