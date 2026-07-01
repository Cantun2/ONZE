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

Bracket structure (general single-elimination)
----------------------------------------------
The base round is read straight from the ``fixtures`` table, ordered by
``(date, fixture_id)``. Its size must be a power of two (e.g. 8 Round-of-16 ties,
or 4 quarter-finals). **Documented linkage assumption:** consecutive base
fixtures feed the same next-round tie — winners of base ties ``2j`` and ``2j+1``
meet in tie ``j`` of the following round — and this pairing repeats up the tree
until one champion remains. This matches how a drawn bracket is laid out in
fixture order.

Venue/home-field: the base round uses each fixture's own ``neutral`` / ``host_flag``
(so a host nation playing at home keeps its Elo bonus, spec §3.1/§7.3). Later
rounds have unknown participants, so they are treated as **neutral** (no home
bonus) — the honest default, since we cannot know in advance whether a host will
reach them or where they will play.

Reproducibility (definition of done)
------------------------------------
The whole simulation is driven by a single seeded ``numpy.random.Generator``
(``seed`` argument, default from a module constant), so results are bit-for-bit
reproducible. Reach/champion probabilities are exact counts / ``n_sims``;
champion probabilities over all teams sum to 1, and the reach columns satisfy
``P(champion) <= P(reach final) <= P(reach semi)``.

Public API
----------
``base_pairs``           read the base-round pairings (+ venue) from the DB.
``simulate_tournament``  Monte-Carlo -> per-team P(reach round) & P(champion).
``main``                 ``python -m src.predict.bracket`` -> print the table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.model.knockout import advance_prob
from src.model.lambdas import predict_lambdas
from src.predict.fixture import _team_elo, load_coeffs
from src.ratings.elo import _home_bonus

DEFAULT_SEED = 20260619  # World Cup 2026 final date, as a stable default seed.

# Venue/context for rounds whose participants are not yet known: neutral ground.
_NEUTRAL_VENUE = {"neutral": True, "host_flag": False}

# Number of base ties -> the label of the round those winners *enter* next.
# Used only for the printed report; the reach columns are keyed off ties-left.
_NEXT_ROUND = {8: "Quarter-final", 4: "Semi-final", 2: "Final", 1: "Champion"}


def base_pairs(con) -> list[tuple[int, int, dict]]:
    """Read the base-round pairings from ``fixtures`` in bracket order.

    The base round is the **widest** round present in ``fixtures`` — the stage
    with the most ties (e.g. the 8 Round-of-16 ties, or 4 quarter-finals). Any
    later-round rows (semi-final / final / third-place) are ignored: their
    participants are decided by the simulation, so only the base pairings and the
    tree shape matter. This lets the sim run whether the seeded fixtures start at
    the Round of 16 or at the quarter-finals.

    Returns a list of ``(home_id, away_id, venue_row)`` tuples ordered by
    ``(date, fixture_id)``; ``venue_row`` carries ``neutral`` / ``host_flag`` for
    the effective home bonus. The number of pairings must be a power of two.
    """
    # Order by fixture_id: it *is* the bracket slot. (Ordering by date would
    # scramble adjacency when a round spans several days, e.g. the Round of 32.)
    fx = con.execute(
        "SELECT fixture_id, date, home_id, away_id, stage, neutral, host_flag "
        "FROM fixtures ORDER BY fixture_id"
    ).fetchdf()
    if fx.empty:
        raise ValueError("fixtures table is empty; nothing to simulate")

    # Widest round = base. Tie-break on the earliest date so the true entry round
    # wins if two stages happen to have equal counts.
    counts = fx.groupby("stage", sort=False)
    base_stage = max(
        counts.groups,
        key=lambda s: (len(counts.groups[s]), -fx.loc[counts.groups[s], "date"].min().toordinal()),
    )
    base = fx[fx["stage"] == base_stage].sort_values("fixture_id")

    pairs = [
        (int(r.home_id), int(r.away_id),
         {"neutral": bool(r.neutral), "host_flag": bool(r.host_flag)})
        for r in base.itertuples(index=False)
    ]
    n = len(pairs)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(
            f"base round '{base_stage}' must have a power-of-two number of ties, found {n}"
        )
    return pairs


def _advance_p_home(
    con, home_id: int, away_id: int, venue_row, c0, c1, rho, cfg: Config
) -> float:
    """``P(home side advances)`` for a tie between ``home_id`` and ``away_id``.

    Wraps the full KO chain: current Elo -> lambdas -> advance_prob (spec §3.5).
    ``venue_row`` supplies ``neutral`` / ``host_flag`` for the effective home bonus.
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
        ``fixtures`` table). Accepted for API compatibility with the spec signature.
    model
        ``model_version`` of the fitted coefficients to use.
    n_sims, seed
        Number of simulations and the RNG seed (reproducible).
    con
        Optional DuckDB connection; opened read-only from config if ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per team in the remaining bracket, columns ``team_id, team,
        p_reach_semi, p_reach_final, p_champion`` — the probability of reaching the
        semi-final, the final, and of winning it. Sorted by ``p_champion`` desc.

    Every value is a Monte-Carlo estimate = count / ``n_sims``. Champion
    probabilities over all teams sum to 1; reach probabilities are monotone
    (champion <= reach_final <= reach_semi). ``reach_semi`` sums to 4 (four
    semi-finalists), ``reach_final`` to 2 — independent of how deep the base round
    is (Round of 16 or quarter-finals).
    """
    if cfg is None:
        cfg = load_config()

    own_con = False
    if con is None:
        import duckdb

        con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
        own_con = True

    try:
        pairs = base_pairs(con)
        c0, c1, rho = load_coeffs(con, model)
        team_names = dict(con.execute("SELECT team_id, name_canonical FROM teams").fetchall())

        all_teams = sorted({t for h, a, _ in pairs for t in (h, a)})
        reach_semi = {t: 0 for t in all_teams}
        reach_final = {t: 0 for t in all_teams}
        champion = {t: 0 for t in all_teams}

        rng = np.random.default_rng(seed)

        # Current round competitors as (n_sims,) arrays; base round is constant.
        homes = [np.full(n_sims, h, dtype=np.int64) for h, a, _ in pairs]
        aways = [np.full(n_sims, a, dtype=np.int64) for h, a, _ in pairs]
        venues = [v for _, _, v in pairs]
        is_base = True

        ties_left = len(pairs)
        while ties_left >= 1:
            # Tally teams that have *reached* this round (its entrants).
            if ties_left == 2:  # this round is the semi-final
                for i in range(2):
                    _tally(reach_semi, homes[i]); _tally(reach_semi, aways[i])
            if ties_left == 1:  # this round is the final
                _tally(reach_final, homes[0]); _tally(reach_final, aways[0])

            winners = []
            for i in range(ties_left):
                venue = venues[i] if is_base else _NEUTRAL_VENUE
                hw = _sim_tie(con, homes[i], aways[i], venue, c0, c1, rho, cfg, rng)
                winners.append(np.where(hw, homes[i], aways[i]))

            if ties_left == 1:
                _tally(champion, winners[0])
                break

            # Pair adjacent winners into the next round (documented linkage).
            homes = [winners[2 * j] for j in range(ties_left // 2)]
            aways = [winners[2 * j + 1] for j in range(ties_left // 2)]
            is_base = False
            ties_left //= 2

        rows = [
            {
                "team_id": t,
                "team": team_names.get(t, str(t)),
                "p_reach_semi": reach_semi[t] / n_sims,
                "p_reach_final": reach_final[t] / n_sims,
                "p_champion": champion[t] / n_sims,
            }
            for t in all_teams
        ]
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
    arrays vary across sims (they are earlier-round winners), so distinct pairings
    appear. We compute ``P(home advances)`` **once per distinct pairing** (via
    :func:`_advance_p_home`, the canonical KO resolver) and draw a Bernoulli
    winner. Returns a boolean array: ``True`` where the home side advanced.
    """
    u = rng.random(home_arr.shape[0])
    pair_keys = list(zip(home_arr.tolist(), away_arr.tolist()))
    cache: dict[tuple[int, int], float] = {}
    for (h, a) in set(pair_keys):
        cache[(h, a)] = _advance_p_home(con, int(h), int(a), venue_row, c0, c1, rho, cfg)
    p_home = np.array([cache[(h, a)] for (h, a) in pair_keys])
    return u < p_home


def main(argv=None) -> None:
    """Run the bracket simulation and print reach/champion probabilities.

    ``python -m src.predict.bracket``. Uses the seeded fixtures, a fixed seed
    (reproducible), and prints the per-team table plus the sanity totals required
    by the definition of done (champion probs sum to 1).
    """
    cfg = load_config()
    df = simulate_tournament(n_sims=100_000, seed=DEFAULT_SEED, cfg=cfg)

    print(f"[bracket] Monte-Carlo tournament sim "
          f"(n_sims={df.attrs['n_sims']:,}, seed={df.attrs['seed']}):\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[bracket] sanity totals (definition of done):")
    print(f"  sum P(champion)     = {df['p_champion'].sum():.6f}   (must be ~1: one champion)")
    print(f"  sum P(reach final)  = {df['p_reach_final'].sum():.6f}   (must be ~2: two finalists)")
    print(f"  sum P(reach semi)   = {df['p_reach_semi'].sum():.6f}   (must be ~4: four semi-finalists)")


if __name__ == "__main__":  # pragma: no cover
    main()
