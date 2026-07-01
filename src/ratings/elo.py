"""World-Football-Elo rating engine for ONZE (spec §3.1, §3.6).

This module owns the Elo ratings layer. It walks the cleaned ``matches`` table
forward in strict chronological order and records, for **every** match, the
rating of each team *as of the instant before kickoff*. Those pre-match ratings
are the only thing downstream models (§3.2 lambdas, §3.7 baselines, backtest)
are ever allowed to see, which is what makes the walk-forward backtest of §3.6
leak-free.

Design contract (do not violate)
--------------------------------
* **No leakage.** The rating stored for a match is computed from all *earlier*
  matches only. A match's own result is applied to the ratings *after* that
  match's pre-match ratings have been recorded. See :func:`compute_elo_history`.
* **Structural constants only.** ``H`` (home bonus), ``base_rating`` and the
  per-match importance weight ``K`` come from ``config.yaml`` /
  ``matches.importance_k``. Nothing here is tuned to fit outcomes; decay and
  hyper-parameters live in the model/eval agents.
* **Deterministic.** Given the same ``matches`` (same rows, same order) the
  output is bit-for-bit reproducible: a stable chronological sort and pure
  arithmetic, no randomness.

Update rules (spec §3.1)
------------------------
Expected home score with the 400-point logistic and an effective home bonus::

    We = 1 / (1 + 10 ** (-(R_home + H_eff - R_away) / 400))

where ``H_eff = H`` when the home team enjoys genuine home advantage and
``H_eff = 0`` on neutral ground. Actual result ``W in {1, 0.5, 0}`` from goals.
Goal-difference multiplier ``G`` (1 / 1.5 / (11+|Δ|)/8). Zero-sum update::

    R'_home = R_home + K * G * (W - We)
    R'_away = R_away - K * G * (W - We)      # sum of ratings conserved

Public API
----------
``compute_elo_history``  Walk-forward, returns the pre-match ratings_history frame.
``elo_1x2``              Bare-Elo 1N2 baseline (spec §3.7 baseline #1).
``rating_as_of``         Leak-free lookup of a team's last rating before a date.
``main``                 ``python -m src.ratings.elo`` end-to-end DB population.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable

import pandas as pd

from src.config import Config, load_config


# ---------------------------------------------------------------------------
# Core Elo maths (spec §3.1)
# ---------------------------------------------------------------------------
def expected_home_score(r_home: float, r_away: float, h_eff: float) -> float:
    """Elo expected score ``We`` for the home team (spec §3.1).

    ``h_eff`` is the *effective* home bonus: ``H`` for a genuine host, ``0`` on
    neutral ground. Ranges in (0, 1); 0.5 means an even match.
    """
    return 1.0 / (1.0 + 10.0 ** (-(r_home + h_eff - r_away) / 400.0))


def goal_diff_multiplier(delta: int) -> float:
    """Goal-difference multiplier ``G`` (spec §3.1).

    ``G = 1`` for a one-goal (or draw) margin, ``1.5`` for a two-goal margin,
    and ``(11 + |Δ|) / 8`` for margins of three or more. Monotone in |Δ|, so a
    heavier win moves the ratings more.
    """
    d = abs(int(delta))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def _result_from_goals(home_goals: int, away_goals: int) -> float:
    """Actual home result ``W in {1, 0.5, 0}`` from the final score."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _home_bonus(neutral: bool, host_flag: bool, H: float) -> float:
    """Effective home bonus for a row.

    History uses ``neutral``: the home team gets ``H`` unless the venue is
    neutral. Fixtures (2026) additionally carry a ``host_flag`` so that a host
    playing at a "neutral" World Cup venue still keeps its real home advantage
    (spec §3.1, §7.3: USA / Canada / Mexico).
    """
    if host_flag:
        return H
    return 0.0 if neutral else H


# ---------------------------------------------------------------------------
# Walk-forward history (spec §3.1 + §3.6 no-leakage)
# ---------------------------------------------------------------------------
def compute_elo_history(
    matches: pd.DataFrame,
    H: float,
    k_by_tournament: dict,
    base_rating: float = 1500.0,
) -> pd.DataFrame:
    """Compute the walk-forward Elo ``ratings_history`` frame.

    Parameters
    ----------
    matches
        Cleaned ``matches`` table (spec §2.2). Required columns:
        ``match_id, date, home_id, away_id, home_goals, away_goals, neutral,
        tournament, importance_k``. An optional ``host_flag`` column is honoured
        if present (fixtures); otherwise host advantage is derived from
        ``neutral`` alone.
    H, base_rating
        Structural Elo constants from ``config.yaml``.
    k_by_tournament
        Fallback map ``tournament -> K`` used only when a row is missing an
        ``importance_k`` value. The per-row ``importance_k`` (already computed by
        the data-engineer) is authoritative.

    Returns
    -------
    pandas.DataFrame
        ``ratings_history`` with columns ``team_id, date, elo`` — two rows per
        played match (home + away), each holding that team's rating **before**
        the match. Sorted by (date, match_id, team_id).

    Notes
    -----
    * **No leakage.** For each match we first *record* both teams' current
      (pre-match) ratings, then apply that match's result. So a stored rating is
      always a function of strictly earlier matches only.
    * **Null goals skipped.** Rows with a missing ``home_goals``/``away_goals``
      (unplayed fixtures, forfeits with no score) are skipped for the *update*
      and produce no history rows; the ratings simply carry forward. This keeps
      future 2026 fixtures from polluting the historical ratings.
    * **Deterministic order.** Matches are sorted by ``(date, match_id)`` with a
      stable sort, so ties on a date resolve reproducibly.
    """
    required = {
        "match_id", "date", "home_id", "away_id",
        "home_goals", "away_goals", "neutral", "tournament", "importance_k",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing required columns: {sorted(missing)}")

    has_host = "host_flag" in matches.columns

    # Strict chronological, deterministic order. mergesort = stable.
    ordered = matches.sort_values(["date", "match_id"], kind="mergesort")

    ratings: dict[int, float] = {}
    hist_team: list[int] = []
    hist_date: list[object] = []
    hist_elo: list[float] = []

    for row in ordered.itertuples(index=False):
        home_id = int(row.home_id)
        away_id = int(row.away_id)

        # Ratings BEFORE this match (base_rating for a team's debut).
        r_home = ratings.get(home_id, base_rating)
        r_away = ratings.get(away_id, base_rating)

        hg = row.home_goals
        ag = row.away_goals
        # Skip unplayed / null-score rows: carry ratings forward, no history row.
        if pd.isna(hg) or pd.isna(ag):
            # Ensure debut teams exist so later lookups are stable.
            ratings.setdefault(home_id, r_home)
            ratings.setdefault(away_id, r_away)
            continue
        hg = int(hg)
        ag = int(ag)

        # --- Record PRE-match ratings (leak-free snapshot) -----------------
        hist_team.append(home_id)
        hist_date.append(row.date)
        hist_elo.append(r_home)
        hist_team.append(away_id)
        hist_date.append(row.date)
        hist_elo.append(r_away)

        # --- Apply the result (spec §3.1) ----------------------------------
        host_flag = bool(getattr(row, "host_flag")) if has_host else False
        h_eff = _home_bonus(bool(row.neutral), host_flag, H)
        we = expected_home_score(r_home, r_away, h_eff)
        w = _result_from_goals(hg, ag)
        g = goal_diff_multiplier(hg - ag)

        k = row.importance_k
        if pd.isna(k):
            k = k_by_tournament.get(row.tournament, k_by_tournament.get("default", 20.0))
        k = float(k)

        change = k * g * (w - we)
        ratings[home_id] = r_home + change
        ratings[away_id] = r_away - change  # zero-sum: total rating conserved

    return pd.DataFrame(
        {"team_id": hist_team, "date": hist_date, "elo": hist_elo}
    )


def latest_ratings(history: pd.DataFrame) -> pd.DataFrame:
    """Each team's most recent rating in ``history`` (columns ``team_id, elo``).

    Note this is the *pre-match* rating of each team's last recorded match, i.e.
    the rating before that match. It is the natural "current strength" estimate
    to seed ``teams.elo_current`` (the post-match value of the very last game is
    not stored, by design, to keep the table strictly leak-free).
    """
    if history.empty:
        return pd.DataFrame({"team_id": [], "elo": []})
    ordered = history.sort_values(["team_id", "date"], kind="mergesort")
    last = ordered.groupby("team_id", as_index=False).last()
    return last[["team_id", "elo"]]


# ---------------------------------------------------------------------------
# Bare-Elo 1N2 baseline (spec §3.7 baseline #1)
# ---------------------------------------------------------------------------
def elo_1x2(
    r_home: float,
    r_away: float,
    h_eff: float,
    draw_factor: float = 0.30,
) -> tuple[float, float, float]:
    """Bare-Elo 1N2 probabilities ``(p_home, p_draw, p_away)`` (spec §3.7 #1).

    This is the honest, model-free baseline the goal model must beat: it derives
    win/draw/loss straight from the Elo expected score ``We``, using *no* goal
    model. There is no single canonical Elo -> draw mapping, so we make ours
    explicit:

    1. ``We`` is the expected home *score* (a win counting 1, a draw 0.5). It is
       the probability of home win plus half the probability of a draw::

           We = p_home + 0.5 * p_draw     (and p_home + p_draw + p_away = 1)

    2. We allocate a draw mass that peaks for even matches and vanishes for
       lopsided ones, via ``p_draw = draw_factor * (1 - |2*We - 1|)``. The
       ``draw_factor`` (default 0.30) sets the draw rate of a coin-flip fixture;
       ~0.30 matches the empirical share of draws in international football and
       is a *baseline* convenience constant, not a tuned model parameter.

    3. The remaining mass splits around ``We`` so that identity (1) holds::

           p_home = We - 0.5 * p_draw
           p_away = (1 - We) - 0.5 * p_draw

    All three are clamped to be non-negative and renormalised, so the result is
    always a valid distribution even at extreme rating gaps.
    """
    we = expected_home_score(r_home, r_away, h_eff)
    p_draw = draw_factor * (1.0 - abs(2.0 * we - 1.0))
    p_home = we - 0.5 * p_draw
    p_away = (1.0 - we) - 0.5 * p_draw

    p_home = max(p_home, 0.0)
    p_away = max(p_away, 0.0)
    p_draw = max(p_draw, 0.0)
    total = p_home + p_draw + p_away
    if total <= 0.0:  # pragma: no cover - defensive, cannot happen for finite Elo
        return (1 / 3, 1 / 3, 1 / 3)
    return (p_home / total, p_draw / total, p_away / total)


# ---------------------------------------------------------------------------
# Leak-free lookup for downstream predictions / backtest (spec §3.6)
# ---------------------------------------------------------------------------
def rating_as_of(
    team_id: int,
    date,
    con=None,
    history: pd.DataFrame | None = None,
    base_rating: float | None = None,
) -> float:
    """Return ``team_id``'s last stored Elo *strictly before* ``date``.

    This is the leak-free lookup every predictor/backtester must use: it never
    returns a rating that incorporates a match on or after ``date``. Supply
    either a DuckDB ``con`` (reads ``ratings_history``) or an in-memory
    ``history`` frame. If the team has no prior recorded match, falls back to
    ``base_rating`` (config default when ``None``).

    ``date`` may be a ``datetime.date``/``datetime`` or an ISO string.
    """
    if base_rating is None:
        base_rating = load_config().base_rating

    if isinstance(date, str):
        date = _dt.date.fromisoformat(date[:10])
    elif isinstance(date, _dt.datetime):
        date = date.date()

    if con is not None:
        res = con.execute(
            """
            SELECT elo FROM ratings_history
            WHERE team_id = ? AND date < ?
            ORDER BY date DESC
            LIMIT 1
            """,
            [int(team_id), date],
        ).fetchone()
        return float(res[0]) if res is not None else float(base_rating)

    if history is None:
        raise ValueError("rating_as_of needs either a `con` or a `history` frame")

    hd = history.copy()
    # Normalise history dates to datetime.date for a clean strict comparison.
    hd_dates = pd.to_datetime(hd["date"]).dt.date
    mask = (hd["team_id"] == int(team_id)) & (hd_dates < date)
    prior = hd.loc[mask]
    if prior.empty:
        return float(base_rating)
    # Last by date (stable: history is emitted in chronological order already).
    idx = pd.to_datetime(prior["date"]).idxmax()
    return float(prior.loc[idx, "elo"])


# ---------------------------------------------------------------------------
# End-to-end DB population (`python -m src.ratings.elo`)
# ---------------------------------------------------------------------------
_MATCH_COLS = [
    "match_id", "date", "home_id", "away_id", "home_goals", "away_goals",
    "neutral", "tournament", "importance_k",
]


def _load_matches(con) -> pd.DataFrame:
    """Read the played-order matches frame from a DuckDB connection."""
    df = con.execute(
        f"SELECT {', '.join(_MATCH_COLS)} FROM matches ORDER BY date, match_id"
    ).fetchdf()
    return df


def _write_history(con, history: pd.DataFrame) -> int:
    """Replace ``ratings_history`` with ``history``; return rows written."""
    con.register("_hist", history)
    con.execute("DELETE FROM ratings_history")
    con.execute("INSERT INTO ratings_history SELECT team_id, date, elo FROM _hist")
    con.unregister("_hist")
    return len(history)


def _update_teams_current(con, history: pd.DataFrame) -> int:
    """Push each team's latest rating into ``teams.elo_current``."""
    latest = latest_ratings(history)
    con.register("_latest", latest)
    con.execute(
        "UPDATE teams SET elo_current = _latest.elo "
        "FROM _latest WHERE teams.team_id = _latest.team_id"
    )
    con.unregister("_latest")
    return len(latest)


def top_n_at_date(
    con,
    as_of: str | _dt.date,
    n: int = 10,
) -> pd.DataFrame:
    """Top-``n`` teams by their last rating strictly before ``as_of`` (sanity check).

    Uses only ``ratings_history`` (leak-free) joined to team names.
    """
    if isinstance(as_of, str):
        as_of = _dt.date.fromisoformat(as_of)
    return con.execute(
        """
        WITH last_before AS (
            SELECT team_id, arg_max(elo, date) AS elo, max(date) AS as_of_date
            FROM ratings_history
            WHERE date < ?
            GROUP BY team_id
        )
        SELECT t.name_canonical AS team, round(lb.elo, 1) AS elo, lb.as_of_date
        FROM last_before lb
        JOIN teams t ON t.team_id = lb.team_id
        ORDER BY lb.elo DESC
        LIMIT ?
        """,
        [as_of, n],
    ).fetchdf()


def main(argv: Iterable[str] | None = None) -> None:
    """Populate ``ratings_history`` + ``teams.elo_current`` and print a sanity check.

    Reads ``matches`` from the configured DuckDB store, computes the full
    walk-forward Elo history, writes it back, refreshes ``teams.elo_current``,
    and prints the top-10 teams at a known past date so a human can eyeball that
    the ranking is plausible.
    """
    import duckdb

    cfg: Config = load_config()
    db_path = str(cfg.paths.db_path)
    con = duckdb.connect(db_path)
    try:
        matches = _load_matches(con)
        history = compute_elo_history(
            matches,
            H=cfg.H,
            k_by_tournament=cfg.k_by_tournament,
            base_rating=cfg.base_rating,
        )
        n_rows = _write_history(con, history)
        n_teams = _update_teams_current(con, history)
        con.commit()

        print(f"[elo] ratings_history rows written: {n_rows:,}")
        print(f"[elo] teams.elo_current updated for {n_teams} teams")
        print(f"[elo] matches read: {len(matches):,} "
              f"(played rows contributing = {n_rows // 2:,})")

        for as_of in ("2018-06-01", "2022-11-01"):
            print(f"\n[elo] Top-10 teams as of {as_of} (rating before that date):")
            top = top_n_at_date(con, as_of, 10)
            print(top.to_string(index=False))
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    main()
