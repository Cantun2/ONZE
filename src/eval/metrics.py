"""Scoring metrics for ONZE evaluation (spec §3.7).

This module is the numerical heart of the evaluation layer: given predicted
probabilities and observed outcomes it returns proper scoring rules. It contains
**no** model or data logic — pure functions over probabilities — so it can be
unit-tested against closed-form worked examples (see ``tests/test_eval.py``).

The three rules (spec §3.7)
---------------------------
* **RPS** — Ranked Probability Score for the *ordered* 1N2 outcome
  ``(home, draw, away)``. This is the primary metric ONZE is judged on. For
  ``r = 3`` ordered categories the spec formula is::

      RPS = (1 / (r - 1)) * sum_{i=1}^{r-1} ( sum_{j<=i} (p_j - e_j) )^2

  with ``p`` the predicted probabilities and ``e`` the one-hot indicator of the
  realised outcome, both taken in the fixed order (home, draw, away). RPS
  rewards putting mass *near* the true ordered outcome, unlike log-loss/Brier
  which ignore the ordering. Lower is better; range ``[0, 1]``.

* **log_loss** — negative log-likelihood for a *binary* market (BTTS, O/U). For
  a single event with predicted P(event) = ``p`` and observed 0/1 ``y``::

      -[ y * ln(p) + (1 - y) * ln(1 - p) ]

* **brier** — squared error for a binary market: ``(p - y)^2``.

All three accept either a single observation or a batch (returning the mean over
the batch), so the backtest can call them once per prediction or once per block.

Public API
----------
``rps``        ordered-1N2 Ranked Probability Score (single or batched).
``log_loss``   binary-market negative log-likelihood (single or batched).
``brier``      binary-market squared error (single or batched).
"""

from __future__ import annotations

import numpy as np

# Fixed ordering of the 1N2 outcome used everywhere in ONZE: home, draw, away.
OUTCOME_ORDER: tuple[str, str, str] = ("home", "draw", "away")
_OUTCOME_INDEX = {"home": 0, "draw": 1, "away": 2, "h": 0, "d": 1, "a": 2}

# Clamp for log-loss so a 0-probability on the realised event does not blow up to
# +inf; standard practice for a bounded, comparable score.
_EPS = 1e-15


# ---------------------------------------------------------------------------
# Ranked Probability Score (spec §3.7, ordered 1N2)
# ---------------------------------------------------------------------------
def _outcome_to_onehot(outcome, r: int = 3) -> np.ndarray:
    """Turn an outcome label/index into a one-hot indicator ``e`` of length ``r``.

    Accepts an integer index (0=home, 1=draw, 2=away), one of the strings in
    :data:`_OUTCOME_INDEX` (``"home"``/``"h"`` etc.), or an already-onehot /
    probability-like array (returned as float, used as ``e``).
    """
    if isinstance(outcome, str):
        idx = _OUTCOME_INDEX.get(outcome.strip().lower())
        if idx is None:
            raise ValueError(f"unknown outcome label {outcome!r}")
        e = np.zeros(r)
        e[idx] = 1.0
        return e
    arr = np.asarray(outcome)
    if arr.ndim == 0:  # scalar index
        idx = int(arr)
        if not 0 <= idx < r:
            raise ValueError(f"outcome index {idx} out of range [0, {r})")
        e = np.zeros(r)
        e[idx] = 1.0
        return e
    if arr.shape[-1] != r:
        raise ValueError(f"outcome array last dim must be {r}, got {arr.shape}")
    return arr.astype(float)


def _rps_single(pred: np.ndarray, e: np.ndarray) -> float:
    """RPS for one prediction/outcome pair, exact spec §3.7 formula.

    RPS = (1 / (r - 1)) * sum_{i=1}^{r-1} ( sum_{j<=i} (p_j - e_j) )^2.
    Uses cumulative sums: the inner ``sum_{j<=i}`` is the cumulative difference,
    and we sum its square over the first ``r - 1`` cumulative positions (the last
    cumulative term is always ~0 since both distributions sum to 1, and the
    formula excludes it).
    """
    r = pred.shape[-1]
    cum = np.cumsum(pred - e)
    # spec sums i = 1..r-1  -> cumulative positions 0..r-2 (drop the final one).
    return float(np.sum(cum[: r - 1] ** 2) / (r - 1))


def rps(pred_probs, outcome) -> float:
    """Ranked Probability Score for the ordered 1N2 (spec §3.7). Lower = better.

    Parameters
    ----------
    pred_probs
        Predicted probabilities in the fixed order ``(home, draw, away)``. Either
        a length-3 vector (single match) or an ``(n, 3)`` array (a batch). Each
        row should sum to ~1; it is **not** renormalised here (callers pass
        proper distributions), but a tiny drift is harmless because the last
        cumulative term is dropped.
    outcome
        The realised outcome, matching the shape of ``pred_probs``:
        for a single match an index (0/1/2) or label (``"home"``/``"draw"``/
        ``"away"``); for a batch a length-``n`` array of indices/labels or an
        ``(n, 3)`` one-hot array.

    Returns
    -------
    float
        The RPS for a single match, or the **mean** RPS over a batch.
    """
    pred = np.asarray(pred_probs, dtype=float)

    if pred.ndim == 1:
        e = _outcome_to_onehot(outcome, r=pred.shape[0])
        return _rps_single(pred, e)

    if pred.ndim == 2:
        r = pred.shape[1]
        # Build the (n, r) one-hot / indicator matrix for the batch.
        if isinstance(outcome, np.ndarray) and outcome.ndim == 2:
            e = outcome.astype(float)
        else:
            e = np.stack([_outcome_to_onehot(o, r=r) for o in outcome])
        cum = np.cumsum(pred - e, axis=1)
        per_match = np.sum(cum[:, : r - 1] ** 2, axis=1) / (r - 1)
        return float(np.mean(per_match))

    raise ValueError(f"pred_probs must be 1-D or 2-D, got ndim={pred.ndim}")


# ---------------------------------------------------------------------------
# Binary-market scoring rules (spec §3.7): log-loss and Brier
# ---------------------------------------------------------------------------
def log_loss(pred_prob, outcome) -> float:
    """Binary-market negative log-likelihood (spec §3.7). Lower = better.

    ``pred_prob`` is P(event) (e.g. P(BTTS) or P(over 2.5)); ``outcome`` is the
    observed 0/1 (or bool). Scalars return that match's loss; equal-length arrays
    return the **mean** log-loss. Probabilities are clamped to ``[eps, 1-eps]`` so
    a certain-but-wrong prediction is heavily but finitely penalised.
    """
    p = np.clip(np.asarray(pred_prob, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(outcome, dtype=float)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(ll))


def brier(pred_prob, outcome) -> float:
    """Binary-market Brier (squared-error) score (spec §3.7). Lower = better.

    ``(p - y)^2`` for P(event) ``p`` and observed 0/1 ``y``. Scalars return that
    match's score; equal-length arrays return the **mean** Brier score. Range
    ``[0, 1]``.
    """
    p = np.asarray(pred_prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))
