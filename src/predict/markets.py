"""Derived betting markets from the single score matrix P(x, y) (spec §3.4).

Every market ONZE exposes for a 90' match is a **deterministic function of the
one object** ``P(X=x, Y=y)`` produced by :func:`src.model.dixon_coles.score_matrix`.
There is no second, ad-hoc model anywhere: 1N2, the most-likely score, the top-5
scores, over/under lines and BTTS are all pure sums over the cells of ``P`` (spec
§0 "on ne prédit jamais un score unique" / §3.4). This is what keeps the whole
output mathematically consistent — the markets can never disagree with the
matrix, because they *are* the matrix, re-summed.

The four §3.4 derivations, reproduced exactly
--------------------------------------------
* **1N2 (90')** — ``P[x, y]`` is ``P(home scores x, away scores y)``::

      p_home = sum_{x>y} P(x, y)     (strictly-lower triangle)
      p_draw = sum_{x=y} P(x, y)     (main diagonal)
      p_away = sum_{x<y} P(x, y)     (strictly-upper triangle)

* **Most-likely / top-5 score** — ``argmax_{(x,y)} P(x, y)`` and the 5 largest.
* **Over/Under line L** — ``over = sum_{x+y > L} P(x, y)``, ``under = 1 - over``.
  Lines are half-integers (0.5 … 4.5) so no cell lands *on* the line.
* **BTTS (both teams to score)** — ``1 - P(X=0) - P(Y=0) + P(0, 0)`` by
  inclusion-exclusion (subtract "home blank" and "away blank", add back the
  double-blank counted twice).

Public API
----------
``derive_markets``   the full §3.4 market dict from a score matrix.
``OVER_UNDER_LINES`` the standard set of O/U lines (0.5 … 4.5).
"""

from __future__ import annotations

import numpy as np

# Standard over/under lines (spec §3.4: "over/under 0.5…4.5"). Half-integers so
# a total of exactly the line is impossible — every scoreline is unambiguously
# over or under.
OVER_UNDER_LINES: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)


def _validate_matrix(P: np.ndarray) -> np.ndarray:
    """Return ``P`` as a square, finite, non-negative 2-D float array.

    ``derive_markets`` assumes ``P`` is a proper (renormalised) pmf as produced
    by :func:`src.model.dixon_coles.score_matrix`; we defensively check shape and
    finiteness so a malformed matrix fails loudly rather than yielding silently
    wrong markets.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"score matrix must be square 2-D, got shape {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("score matrix contains non-finite entries")
    if (P < 0).any():
        raise ValueError("score matrix contains negative probabilities")
    return P


def one_x_two(P: np.ndarray) -> tuple[float, float, float]:
    """1N2 probabilities ``(p_home, p_draw, p_away)`` summed from ``P`` (spec §3.4).

    ``p_home = sum_{x>y}`` (lower triangle), ``p_draw = sum_{x=y}`` (diagonal),
    ``p_away = sum_{x<y}`` (upper triangle). These three partition the matrix, so
    they sum to ``P.sum()`` (== 1 for a renormalised score matrix).
    """
    p_home = float(np.tril(P, -1).sum())  # x > y
    p_draw = float(np.trace(P))           # x == y
    p_away = float(np.triu(P, 1).sum())   # x < y
    return p_home, p_draw, p_away


def most_likely_score(P: np.ndarray) -> tuple[int, int]:
    """The single most probable scoreline ``argmax_{(x,y)} P(x, y)`` (spec §3.4)."""
    x, y = np.unravel_index(int(np.argmax(P)), P.shape)
    return int(x), int(y)


def top_n_scores(P: np.ndarray, n: int = 5) -> list[tuple[tuple[int, int], float]]:
    """The ``n`` most probable scorelines as ``[((x, y), prob), ...]`` (spec §3.4).

    Sorted by descending probability. Ties on probability are broken by the
    natural (x, y) order for reproducibility.
    """
    flat = P.ravel()
    n = min(n, flat.size)
    # argpartition gets the top-n cheaply; then sort those n by prob desc, with a
    # stable (x, y) tiebreak so the output is deterministic.
    idx = np.argpartition(flat, -n)[-n:]
    scores = []
    for i in idx:
        x, y = np.unravel_index(int(i), P.shape)
        scores.append(((int(x), int(y)), float(flat[i])))
    scores.sort(key=lambda s: (-s[1], s[0]))
    return scores


def over_under(P: np.ndarray, lines=OVER_UNDER_LINES) -> dict[str, dict[str, float]]:
    """Over/Under probabilities for each ``line`` (spec §3.4).

    ``over = sum_{x+y > line} P(x, y)``, ``under = 1 - over`` (equivalently
    ``sum_{x+y < line}`` since lines are half-integers). Keyed by the line as a
    string, e.g. ``"2.5"``, each value ``{"over": ..., "under": ...}``.
    """
    K = P.shape[0] - 1
    # totals[t] = P(X + Y = t) for t in 0..2K, via the anti-diagonal sums.
    totals = np.array([np.trace(P[:, ::-1], offset=K - t) for t in range(2 * K + 1)])
    out: dict[str, dict[str, float]] = {}
    for line in lines:
        # x + y > line  <=>  total >= ceil(line) = line + 0.5 (half-integer line).
        over = float(totals[int(np.ceil(line)):].sum())
        out[f"{line:g}"] = {"over": over, "under": 1.0 - over}
    return out


def btts(P: np.ndarray) -> float:
    """P(both teams score) = ``1 - P(X=0) - P(Y=0) + P(0,0)`` (spec §3.4).

    Inclusion-exclusion: subtract the "home fails to score" row and the "away
    fails to score" column, then add back ``P(0, 0)`` which both removed.
    """
    p_home_blank = float(P[0, :].sum())   # X = 0
    p_away_blank = float(P[:, 0].sum())   # Y = 0
    p_double_blank = float(P[0, 0])       # X = 0 and Y = 0
    return 1.0 - p_home_blank - p_away_blank + p_double_blank


def derive_markets(P: np.ndarray) -> dict:
    """All §3.4 derived markets from the score matrix ``P`` — no separate model.

    Returns a dict with:

    * ``p_home`` / ``p_draw`` / ``p_away`` — 1N2 at 90' (partition of ``P``);
    * ``most_likely_score`` — ``(x, y)`` tuple and ``"x-y"`` string;
    * ``top5_scores`` — ``[((x, y), prob), ...]`` five most-likely scorelines;
    * ``over_under`` — ``{ "0.5": {over, under}, ..., "4.5": {...} }``;
    * ``btts`` — P(both teams score).

    ``P[x, y] = P(home scores x, away scores y)``. Every value is a pure sum over
    ``P``, so the markets are mutually consistent by construction (spec §0/§3.4).
    """
    P = _validate_matrix(P)

    p_home, p_draw, p_away = one_x_two(P)
    x_ml, y_ml = most_likely_score(P)
    top5 = top_n_scores(P, 5)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "most_likely_score": (x_ml, y_ml),
        "most_likely_score_str": f"{x_ml}-{y_ml}",
        "top5_scores": top5,
        "over_under": over_under(P),
        "btts": btts(P),
    }
