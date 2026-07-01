"""Monte-Carlo simulation of the remaining knockout bracket (spec §1, §5).

The single-match layer gives, for any pairing, the probability each side
advances (:func:`src.model.knockout.advance_prob`, spec §3.5). To turn that into
tournament outputs — ``P(reach quarter / semi / final)`` and ``P(champion)`` per
team (spec §1, §5 view 3) — we simulate the *actual remaining bracket tree* many
times and count how often each team reaches each round.

How a tie is resolved in a simulation (documented choice)
--------------------------------------------------------
For every simulated tie we draw the winner from a Bernoulli with success
probability ``advance_prob(...)["p_home_advance"]`` — i.e. the *already
KO-resolved* qualification probability that folds in 90' + extra time +
penalties (spec §3.5). We do **not** sample a raw scoreline and re-resolve it by
hand: ``advance_prob`` is knockout-engineer's canonical resolver, so sampling its
output is exactly consistent with the per-fixture qualification numbers
``fixture.py`` reports. This keeps one source of truth for "who goes through".
(The alternative — sampling a scoreline from ``P`` then re-resolving ET/pens — has
the same expectation but adds Monte-Carlo noise for no modelling gain, so we
avoid it.)

Bracket structure (the 8 illustrative 2026 fixtures)
---------------------------------------------------
The seeded fixtures form a standard QF -> SF -> Final tree plus a third-place
play-off. We encode the linkage explicitly in :data:`BRACKET` below. **Documented
assumption:** the four quarter-finals feed the two semi-finals in fixture order —
QF0 & QF1 winners meet in SF0, QF2 & QF3 winners meet in SF1 — and the two SF
winners meet in the Final; the two SF losers meet in the third-place play-off.
The seeded SF/Final rows name specific teams only illustratively (one possible
outcome); the simulation replaces those with the actual simulated winners, so the
seeded home/away teams of SF/Final rows are ignored for linkage — only their
fixture ``date``, ``stage``, ``neutral`` and ``host_flag`` (venue/context) are
used.

Reproducibility (definition of done)
------------------------------------
The whole simulation is driven by a single seeded ``numpy.random.Generator``
(``seed`` argument, default from a module constant), so results are bit-for-bit
reproducible. Per-round reach-probabilities and champion probabilities are exact
counts / ``n_sims``; champion probabilities over all teams sum to 1.

Public API
----------
``build_bracket``        assemble the bracket + per-tie advance probs from the DB.
``simulate_tournament``  Monte-Carlo -> per-team P(reach round) & P(champion).
``main``                 ``python -m src.predict.bracket`` -> print the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.model.knockout import advance_prob
from src.model.lambdas import predict_lambdas
from src.predict.fixture import _team_elo, load_coeffs
from src.ratings.elo import _home_bonus

DEFAULT_SEED = 20260619  # World Cup 2026 final date, as a stable default seed.


# ---------------------------------------------------------------------------
# Bracket linkage for the seeded 2026 fixtures (documented in the module header)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BracketNode:
    """One tie in the bracket tree.

    A node's two competitors are either concrete teams (quarter-finals, whose
    participants are already known) or the *winner*/*loser* of earlier nodes.
    ``feeds`` maps a slot ("home"/"away") to either ``("team", team_id)`` or
    ``("winner", node_key)`` / ``("loser", node_key)``.
    """

    key: str
    round_name: str          # human round label ("Quarter-final", ...)
    feeds: dict[str, tuple]  # "home"/"away" -> ("team", id) | ("winner"|"loser", key)


# The remaining-round order a team can reach, coarsest last. Used to accumulate
# "reached at least this round" counts. (A quarter-finalist has, by definition,
# already reached the quarter-finals — that is the entry round of this bracket.)
ROUND_ORDER = ["Quarter-final", "Semi-final", "Final", "Champion"]


def build_bracket(
    con,
    model_version: str = "dc-v1",
    cfg: Config | None = None,
) -> tuple[dict[str, BracketNode], dict[str, float], dict[int, str]]:
    """Assemble the bracket tree and each *quarter-final's* advance probability.

    Returns ``(nodes, qf_p_home_advance, team_names)`` where:

    * ``nodes`` maps node-key -> :class:`BracketNode` for QFs, SFs, the Final and
      the third-place play-off, with the linkage described in the module header;
    * ``qf_p_home_advance`` maps each QF node-key -> ``P(home side advances)``
      from :func:`src.model.knockout.advance_prob` (the QF pairings are known, so
      their advance probs are fixed up front). SF/Final/3rd-place advance probs
      depend on who wins earlier rounds, so they are computed *inside* the
      simulation per draw (see :func:`_advance_p_home`).
    * ``team_names`` maps team_id -> canonical name.

    Reads current Elo (``teams.elo_current``, honouring neutral/host_flag) and the
    fitted coefficients; does not modify any upstream module.
    """
    if cfg is None:
        cfg = load_config()

    fx = con.execute(
        "SELECT fixture_id, date, home_id, away_id, stage, neutral, host_flag "
        "FROM fixtures ORDER BY date, fixture_id"
    ).fetchdf()

    c0, c1, rho = load_coeffs(con, model_version)

    # Split by stage. The 4 quarter-finals (known pairings) drive the tree base.
    qfs = fx[fx["stage"].str.lower() == "quarter-final"].reset_index(drop=True)
    sfs = fx[fx["stage"].str.lower() == "semi-final"].reset_index(drop=True)
    third = fx[fx["stage"].str.lower() == "third place"].reset_index(drop=True)
    final = fx[fx["stage"].str.lower() == "final"].reset_index(drop=True)

    if len(qfs) != 4:
        raise ValueError(f"expected 4 quarter-finals, found {len(qfs)}")

    nodes: dict[str, BracketNode] = {}
    qf_p_home_advance: dict[str, float] = {}

    # --- Quarter-finals: concrete pairings, fixed advance probs ---------------
    for i, r in qfs.iterrows():
        key = f"QF{i}"
        home_id, away_id = int(r["home_id"]), int(r["away_id"])
        nodes[key] = BracketNode(
            key=key,
            round_name="Quarter-final",
            feeds={"home": ("team", home_id), "away": ("team", away_id)},
        )
        qf_p_home_advance[key] = _advance_p_home(
            con, home_id, away_id, r, c0, c1, rho, cfg
        )

    # --- Semi-finals: winners of QF pairs, in fixture order (documented) -------
    # SF0 = winner(QF0) vs winner(QF1); SF1 = winner(QF2) vs winner(QF3).
    sf_venue = _venue_rows(sfs, 2)
    nodes["SF0"] = BracketNode(
        key="SF0", round_name="Semi-final",
        feeds={"home": ("winner", "QF0"), "away": ("winner", "QF1")},
    )
    nodes["SF1"] = BracketNode(
        key="SF1", round_name="Semi-final",
        feeds={"home": ("winner", "QF2"), "away": ("winner", "QF3")},
    )

    # --- Final: winners of the two semi-finals --------------------------------
    nodes["FINAL"] = BracketNode(
        key="FINAL", round_name="Final",
        feeds={"home": ("winner", "SF0"), "away": ("winner", "SF1")},
    )

    # --- Third place: losers of the two semi-finals (not a route to champion) --
    nodes["THIRD"] = BracketNode(
        key="THIRD", round_name="Third place",
        feeds={"home": ("loser", "SF0"), "away": ("loser", "SF1")},
    )

    # Venue/context rows for the later nodes (team identities are ignored; only
    # neutral/host_flag/date drive H_eff for those ties).
    nodes_venue = {
        "SF0": sf_venue[0], "SF1": sf_venue[1],
        "FINAL": _venue_rows(final, 1)[0],
        "THIRD": _venue_rows(third, 1)[0],
    }

    team_names = dict(con.execute("SELECT team_id, name_canonical FROM teams").fetchall())

    # Stash coeffs + venue on a lightweight context the simulator reuses.
    _BRACKET_CTX["coeffs"] = (c0, c1, rho)
    _BRACKET_CTX["venue"] = nodes_venue
    _BRACKET_CTX["cfg"] = cfg

    return nodes, qf_p_home_advance, team_names


# Simulation context populated by build_bracket (coeffs + per-node venue rows).
_BRACKET_CTX: dict[str, Any] = {}


def _venue_rows(df: pd.DataFrame, n_expected: int) -> list[dict]:
    """Return ``n_expected`` venue/context rows (neutral/host_flag/...) as dicts.

    If the seeded set is short (illustrative fixtures), pad with a neutral World
    Cup venue so the simulation still runs.
    """
    rows = df.to_dict("records")
    while len(rows) < n_expected:
        rows.append({"neutral": True, "host_flag": False})
    return rows[:n_expected]


def _advance_p_home(
    con, home_id: int, away_id: int, venue_row, c0, c1, rho, cfg: Config
) -> float:
    """``P(home side advances)`` for a tie between ``home_id`` and ``away_id``.

    Wraps the full KO chain: current Elo -> lambdas -> advance_prob (spec §3.5).
    ``venue_row`` supplies ``neutral`` / ``host_flag`` for the effective home
    bonus (a mapping or a namedtuple-like row).
    """
    neutral = bool(_get(venue_row, "neutral", True))
    host_flag = bool(_get(venue_row, "host_flag", False))
    h_eff = _home_bonus(neutral, host_flag, cfg.H)

    elo_home = _team_elo(con, home_id)
    elo_away = _team_elo(con, away_id)
    lam, mu = predict_lambdas(elo_home, elo_away, h_eff, c0, c1)
    adv = advance_prob(
        lam, mu, rho, elo_home, elo_away,
        theta=cfg.theta, kappa=cfg.kappa, K=cfg.K_max,
    )
    return adv["p_home_advance"]


def _get(row, key, default):
    """Read ``key`` from a dict-or-attr row, with a default."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


# ---------------------------------------------------------------------------
# Monte-Carlo over the bracket tree (spec §1, §5) — seeded / reproducible
# ---------------------------------------------------------------------------
def simulate_tournament(
    remaining_fixtures=None,
    model: str = "dc-v1",
    n_sims: int = 100_000,
    seed: int = DEFAULT_SEED,
    con=None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Monte-Carlo the remaining bracket -> per-team reach & champion probs.

    Parameters
    ----------
    remaining_fixtures
        Ignored when reading straight from the DB (the bracket is built from the
        ``fixtures`` table). Accepted for API compatibility with the spec
        signature; pass ``None`` to use the seeded fixtures.
    model
        ``model_version`` of the fitted coefficients to use.
    n_sims, seed
        Number of simulations and the RNG seed (reproducible).
    con
        Optional DuckDB connection; opened read-only from config if ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per team that is in the remaining bracket, columns:
        ``team_id, team, p_reach_semi, p_reach_final, p_champion``. (A
        quarter-finalist has already reached the quarter-finals, so that column
        is trivially 1 and omitted; the interesting remaining rounds are semi,
        final and champion — spec §1/§5.) Sorted by ``p_champion`` descending.

    Every value is a Monte-Carlo estimate = count / ``n_sims``. Champion
    probabilities over all teams sum to 1 (exactly one champion per sim); reach
    probabilities are monotone (champion <= reach_final <= reach_semi).
    """
    if cfg is None:
        cfg = load_config()

    own_con = False
    if con is None:
        import duckdb

        con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
        own_con = True

    try:
        nodes, qf_p_home, team_names = build_bracket(con, model, cfg)
        c0, c1, rho = _BRACKET_CTX["coeffs"]
        venue = _BRACKET_CTX["venue"]

        # Teams entering the bracket (the 8 quarter-finalists).
        qf_teams: list[int] = []
        for k in ("QF0", "QF1", "QF2", "QF3"):
            qf_teams.append(nodes[k].feeds["home"][1])
            qf_teams.append(nodes[k].feeds["away"][1])

        rng = np.random.default_rng(seed)

        # Vectorised draws for the four QFs up front (fixed advance probs).
        qf_home_wins = {
            k: rng.random(n_sims) < qf_p_home[k] for k in ("QF0", "QF1", "QF2", "QF3")
        }

        # Counters: reached-semi / reached-final / champion, per team.
        reach_semi = {t: 0 for t in qf_teams}
        reach_final = {t: 0 for t in qf_teams}
        champion = {t: 0 for t in qf_teams}

        # Winners of each QF as team-id arrays over the n_sims draws.
        def qf_winner(k: str) -> np.ndarray:
            home = nodes[k].feeds["home"][1]
            away = nodes[k].feeds["away"][1]
            return np.where(qf_home_wins[k], home, away)

        qf0_w, qf1_w = qf_winner("QF0"), qf_winner("QF1")
        qf2_w, qf3_w = qf_winner("QF2"), qf_winner("QF3")

        # SF0 = qf0_w vs qf1_w ; SF1 = qf2_w vs qf3_w. Both QF winners have
        # reached the semi-finals.
        for arr in (qf0_w, qf1_w, qf2_w, qf3_w):
            _tally(reach_semi, arr)

        sf0_home_wins = _sim_tie(con, qf0_w, qf1_w, venue["SF0"], c0, c1, rho, cfg, rng)
        sf1_home_wins = _sim_tie(con, qf2_w, qf3_w, venue["SF1"], c0, c1, rho, cfg, rng)

        sf0_w = np.where(sf0_home_wins, qf0_w, qf1_w)
        sf1_w = np.where(sf1_home_wins, qf2_w, qf3_w)

        # Both SF winners have reached the Final.
        _tally(reach_final, sf0_w)
        _tally(reach_final, sf1_w)

        # Final: sf0_w vs sf1_w -> champion.
        final_home_wins = _sim_tie(con, sf0_w, sf1_w, venue["FINAL"], c0, c1, rho, cfg, rng)
        champ = np.where(final_home_wins, sf0_w, sf1_w)
        _tally(champion, champ)

        rows = []
        for t in qf_teams:
            rows.append(
                {
                    "team_id": t,
                    "team": team_names.get(t, str(t)),
                    "p_reach_semi": reach_semi[t] / n_sims,
                    "p_reach_final": reach_final[t] / n_sims,
                    "p_champion": champion[t] / n_sims,
                }
            )
        df = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
        df.attrs["n_sims"] = n_sims
        df.attrs["seed"] = seed
        return df
    finally:
        if own_con:
            con.close()


def _tally(counter: dict[int, int], team_arr: np.ndarray) -> None:
    """Add per-team occurrence counts from ``team_arr`` into ``counter``."""
    ids, counts = np.unique(team_arr, return_counts=True)
    for tid, c in zip(ids.tolist(), counts.tolist()):
        counter[tid] += int(c)


def _sim_tie(
    con, home_arr: np.ndarray, away_arr: np.ndarray, venue_row,
    c0, c1, rho, cfg: Config, rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a whole round of ties given the two competitor arrays.

    ``home_arr[i]`` vs ``away_arr[i]`` is the tie in simulation ``i``. The two
    arrays vary across sims (they are earlier-round winners), so distinct
    pairings appear. We compute ``P(home advances)`` **once per distinct pairing**
    (via :func:`_advance_p_home`, the canonical KO resolver) and draw a Bernoulli
    winner. Returns a boolean array: ``True`` where the home side advanced.
    """
    home_wins = np.empty(home_arr.shape[0], dtype=bool)
    u = rng.random(home_arr.shape[0])

    # Cache advance probs per (home_id, away_id) pairing to avoid recomputation.
    cache: dict[tuple[int, int], float] = {}
    # Group identical pairings for a vectorised comparison.
    pair_keys = list(zip(home_arr.tolist(), away_arr.tolist()))
    unique_pairs = set(pair_keys)
    for (h, a) in unique_pairs:
        if (h, a) not in cache:
            cache[(h, a)] = _advance_p_home(con, int(h), int(a), venue_row, c0, c1, rho, cfg)

    p_home = np.array([cache[(h, a)] for (h, a) in pair_keys])
    home_wins = u < p_home
    return home_wins


def main(argv=None) -> None:
    """Run the bracket simulation and print reach/champion probabilities.

    ``python -m src.predict.bracket``. Uses the seeded 2026 fixtures, a fixed
    seed (reproducible), and prints the per-team table plus the sanity totals
    required by the definition of done (champion probs sum to 1).
    """
    cfg = load_config()
    df = simulate_tournament(n_sims=100_000, seed=DEFAULT_SEED, cfg=cfg)

    print(f"[bracket] Monte-Carlo tournament sim "
          f"(n_sims={df.attrs['n_sims']:,}, seed={df.attrs['seed']}):\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    p_champ_sum = df["p_champion"].sum()
    p_final_sum = df["p_reach_final"].sum()
    p_semi_sum = df["p_reach_semi"].sum()
    print("\n[bracket] sanity totals (definition of done):")
    print(f"  sum P(champion)     = {p_champ_sum:.6f}   (must be ~1: one champion)")
    print(f"  sum P(reach final)  = {p_semi_sum and p_final_sum:.6f}   (must be ~2: two finalists)")
    print(f"  sum P(reach semi)   = {p_semi_sum:.6f}   (must be ~4: four semi-finalists)")


if __name__ == "__main__":  # pragma: no cover
    main()
