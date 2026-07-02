"""Reliability diagrams and calibration error for ONZE (spec §3.7).

A model can have a good RPS yet be mis-*calibrated*: when it says "70%", does the
event happen 70% of the time? This module answers that. It bins predictions by
predicted probability, compares the mean predicted probability against the
observed frequency in each bin (the reliability diagram), and summarises the gap
as an Expected Calibration Error (ECE). A perfectly calibrated model sits on the
diagonal (mean-predicted == observed) in every bin.

What it operates on
-------------------
A flat pair of arrays ``(pred, obs)`` where each ``pred[k]`` is a predicted
probability of a **binary** event and ``obs[k]`` is the realised 0/1. For the
1N2 case the natural flattening is "one row per (match, outcome-class)": each
match contributes three rows — P(home) vs 1{home}, P(draw) vs 1{draw},
P(away) vs 1{away} — which is exactly how :func:`onehot_flatten` prepares it.
This one-vs-rest flattening is the standard way to read a multi-class model's
calibration off a single reliability curve.

Outputs
-------
* :func:`reliability_table` — a bin-by-bin :class:`pandas.DataFrame`
  (bin edges, count, mean predicted, observed frequency, gap).
* :func:`expected_calibration_error` — the count-weighted mean absolute gap.
* :func:`render` — best-effort matplotlib PNG to ``reports/``; if matplotlib is
  **absent** (guarded import), it degrades to writing the table as CSV +
  markdown so the report is always produced.

Public API
----------
``onehot_flatten``               1N2 preds/outcomes -> one-vs-rest (pred, obs).
``reliability_table``            binned calibration table.
``expected_calibration_error``   ECE summary (count-weighted |gap|).
``render``                       write PNG (if matplotlib) or CSV/markdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Flattening 1N2 predictions into a one-vs-rest binary calibration problem
# ---------------------------------------------------------------------------
def onehot_flatten(pred_probs, outcomes) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ordered-1N2 predictions to one-vs-rest ``(pred, obs)`` arrays.

    ``pred_probs`` is ``(n, 3)`` in order (home, draw, away); ``outcomes`` is a
    length-``n`` array of realised class indices (0/1/2). Returns two length-3n
    arrays: every predicted class-probability paired with the 0/1 indicator of
    whether that class actually occurred. This is what feeds the reliability
    table so a single curve summarises the whole 1N2 model's calibration.
    """
    pred = np.asarray(pred_probs, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred_probs must be (n, 3), got {pred.shape}")
    out = np.asarray(outcomes)
    n, r = pred.shape
    onehot = np.zeros((n, r))
    onehot[np.arange(n), out.astype(int)] = 1.0
    return pred.reshape(-1), onehot.reshape(-1)


# ---------------------------------------------------------------------------
# Reliability table (spec §3.7 "diagramme de fiabilité")
# ---------------------------------------------------------------------------
def reliability_table(pred, obs, n_bins: int = 10) -> pd.DataFrame:
    """Bin ``(pred, obs)`` and compare mean predicted vs observed frequency.

    Predictions are grouped into ``n_bins`` equal-width bins over ``[0, 1]``.
    For each **non-empty** bin the table reports the bin range, the number of
    predictions, the mean predicted probability, the observed event frequency,
    and their signed gap ``(mean_pred - observed)``. A well-calibrated model has
    ``gap ~ 0`` in every bin (points on the diagonal).

    Returns a DataFrame with columns
    ``[bin_lo, bin_hi, count, mean_pred, observed, gap]``.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"pred {pred.shape} and obs {obs.shape} must match")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each prediction to a bin; clip so p == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mp = float(pred[mask].mean())
        ob = float(obs[mask].mean())
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": cnt,
                "mean_pred": mp,
                "observed": ob,
                "gap": mp - ob,
            }
        )
    return pd.DataFrame(
        rows, columns=["bin_lo", "bin_hi", "count", "mean_pred", "observed", "gap"]
    )


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Expected Calibration Error: count-weighted mean absolute bin gap.

    ``ECE = sum_b (count_b / N) * |mean_pred_b - observed_b|``. 0 is perfect;
    typical well-calibrated football models land around 0.01–0.03. Computed from
    a :func:`reliability_table`.
    """
    if table.empty:
        return float("nan")
    n = table["count"].sum()
    if n == 0:
        return float("nan")
    return float((table["count"] * table["gap"].abs()).sum() / n)


# ---------------------------------------------------------------------------
# Rendering: matplotlib PNG if available, else CSV + markdown (guarded import)
# ---------------------------------------------------------------------------
def render(
    table: pd.DataFrame,
    out_dir: str | Path,
    stem: str = "calibration",
    title: str | None = None,
) -> dict[str, Path]:
    """Persist the reliability diagram. PNG if matplotlib is present, else CSV+MD.

    Always writes ``<stem>.csv`` (the raw table). If matplotlib imports cleanly
    it additionally renders ``<stem>.png`` (reliability curve vs the diagonal);
    otherwise it writes a ``<stem>.md`` markdown rendering of the table so the
    calibration result is human-readable without a plotting backend. Returns a
    dict of the artefact name -> path actually written.

    The matplotlib import is **guarded** (try/except) exactly because the ONZE
    runtime image may not ship a plotting backend; calibration must never be the
    thing that breaks a headless CI run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    csv_path = out_dir / f"{stem}.csv"
    table.to_csv(csv_path, index=False)
    written["csv"] = csv_path

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, no display needed
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        if not table.empty:
            ax.plot(
                table["mean_pred"],
                table["observed"],
                "o-",
                color="C0",
                label="model",
            )
            # marker size ~ bin count, so sparse bins are visually de-emphasised.
            sizes = 20.0 + 180.0 * table["count"] / table["count"].max()
            ax.scatter(
                table["mean_pred"], table["observed"], s=sizes, alpha=0.3, color="C0"
            )
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title or "Reliability diagram")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        written["png"] = png_path
    except Exception:
        # matplotlib absent (or backend failure): degrade to a markdown table.
        md_path = out_dir / f"{stem}.md"
        header = title or "Reliability diagram"
        lines = [
            f"# {header}",
            "",
            "_(matplotlib unavailable — table fallback)_",
            "",
            "| bin | count | mean_pred | observed | gap |",
            "|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| [{row['bin_lo']:.2f}, {row['bin_hi']:.2f}) "
                f"| {int(row['count'])} | {row['mean_pred']:.3f} "
                f"| {row['observed']:.3f} | {row['gap']:+.3f} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["md"] = md_path

    return written
