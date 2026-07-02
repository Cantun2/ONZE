# ONZE — a score-distribution engine for the 2026 World Cup

ONZE does **not** predict a single scoreline. For every remaining match it
produces a full probability matrix **P(x, y)** — the probability that the home
team scores `x` and the away team scores `y` — and derives *everything else*
(1N2, over/under, both-teams-to-score, qualification, champion odds) from that
one object. The matrix is the only mathematically honest unit of output, and it
is also the richest thing to look at.

> **Honesty framing (read this first).** ONZE is a distribution engine, not a
> tipster. The outputs are probability distributions. Guessing the *exact* final
> score is intrinsically hard: an exact-score hit rate of **~10–15% is the
> ceiling, not a failure** — football is that random. The "most-likely score" the
> UI shows is just the *mode* of the distribution, usually well under 20% likely.
> The honest bar to judge the model against is the **de-vigged bookmaker RPS**
> (matching it is an excellent result), and **calibration** (does "70%" happen
> 70% of the time). There are no "beat the bookmaker / get rich" claims here.

---

## What it is

A reproducible pipeline that turns 150 years of international match results into
per-match score distributions and tournament simulations, served through a
FastAPI backend and a React interface.

### The value chain

```
Historical results ─► Elo ratings (as-of each date) ─► Dixon-Coles goals model
   (martj42 +                (walk-forward,               (Elo diff → λ, μ ;
    Fjelstul)                 leak-free)                   low-score τ correction)
                                                                   │
                                                                   ▼
                                        Score matrix  P(x, y)  per match
                                                                   │
             ┌──────────────────┬──────────────────┬──────────────┴───────────┐
             ▼                  ▼                  ▼                            ▼
        Markets 1N2        Most-likely        Knockout module            Monte-Carlo
        O/U, BTTS          score (mode)       (90'+ET+penalties)         bracket sim
                                              P(qualify)                 P(champion)
                                                                   │
                                                                   ▼
                             Evaluation (RPS / log-loss / calibration vs baselines)
                                                                   │
                                                                   ▼
                                        FastAPI  ─►  React + Vite interface
```

Everything downstream of the matrix is a *pure sum over its cells* — the markets
can never disagree with the matrix, because they **are** the matrix, re-summed.

---

## Quick start

### Requirements

- **Python 3.11+** (the pinned dependency set is validated on 3.12).
- **Node 22+ / npm** for the frontend.

### Install (Python)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All the `make` targets below assume this virtualenv. Either activate it
(`source .venv/bin/activate`) or run the commands with `PYTHON=.venv/bin/python`.

### Run the pipeline (in ≤ 5 commands)

The offline pipeline is four stages, then serve:

```bash
make ingest      # 1. load martj42 + Fjelstul, harmonise team names -> DuckDB
make ratings     # 2. Elo walk-forward -> leak-free ratings_history
make fit         # 3. fit the Dixon-Coles goal model (c0, c1, rho)
make backtest    # 4. walk-forward backtest vs baselines -> reports/
make api         # 5. serve the FastAPI app on http://localhost:8000
```

`make all` runs steps 1–4 in order (`ingest → ratings → fit → backtest`). After
`make fit` you can also populate the per-fixture predictions table with
`.venv/bin/python -m src.predict.fixture`.

The backtest is a *smoke* window by default when run for CI; for the full
multi-decade run use `.venv/bin/python -m src.eval.backtest` (no `--smoke`).

### Run the frontend

```bash
make frontend    # cd frontend && npm run dev  (Vite dev server on :5173)
```

The frontend has its own setup notes and env vars in
[`frontend/README.md`](frontend/README.md). It consumes the API read-only; make
sure `make api` is running.

### Tests

```bash
make test        # python -m pytest
```

---

## What the numbers look like

From the committed smoke backtest (2024-01-01 → 2026-06-30, 2,623 matches; see
[`reports/BACKTEST_REPORT.md`](reports/BACKTEST_REPORT.md)):

| Predictor                    | Mean RPS (lower is better) |
| ---------------------------- | -------------------------- |
| **ONZE model (Dixon-Coles)** | **0.16844** (beats Elo)    |
| Bare Elo baseline            | 0.17040                    |
| De-vigged bookmaker          | *pending odds ingestion*   |

Calibration on the same window: **ECE ≈ 0.017** (0 is perfect). The model beats
the bare-Elo baseline on RPS, which shows the goal model adds value over the
rating alone. The bookmaker baseline — the *real* bar — is guarded and activates
automatically once closing odds are loaded into the `odds` table.

> The remaining ~30 live 2026 World Cup matches are a **demo, not a validation
> set**: 30 matches cannot separate models. The statistically meaningful
> validation is the multi-decade walk-forward backtest above.

---

## How it works (the short version)

| Stage             | Module                        | What it does                                                                              |
| ----------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| **Data**          | `src/ingest/`                 | Load martj42 (~49,496 matches, 1872–2026) + Fjelstul; harmonise to **336 canonical teams**, **0 unresolved names**. |
| **Ratings**       | `src/ratings/elo.py`          | World-Football-Elo, walked forward match-by-match; stores each team's **pre-match** rating (leak-free). |
| **Goals model**   | `src/model/lambdas.py`, `dixon_coles.py` | Map Elo differential → expected goals (λ, μ); Dixon-Coles low-score τ correction; time-decayed MLE of ρ. |
| **Knockout**      | `src/model/knockout.py`       | 90' + extra time + penalty shootout → P(qualify); the two sides' advance probs sum to 1.   |
| **Predict**       | `src/predict/`                | Per-fixture pipeline, derived markets, and a Monte-Carlo bracket simulation.               |
| **Eval**          | `src/eval/`                   | RPS / log-loss / Brier, reliability diagrams, strict walk-forward backtest with date assertions. |
| **API**           | `src/api/main.py`             | FastAPI, 5 endpoints (thin serving layer over precomputed artifacts).                      |
| **Frontend**      | `frontend/`                   | React + Vite, 4 views including the `P(x, y)` heatmap.                                     |

For the full mathematics — Elo update rules, the Elo→λ regression, the
Dixon-Coles likelihood, the knockout decomposition, the no-leakage protocol and
the evaluation metrics — see **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)**.
For a plain-language guide to reading the interface, see
**[`docs/READING_OUTPUTS.md`](docs/READING_OUTPUTS.md)**.

---

## Project layout

```
ONZE/
├── config.yaml              # single source of truth for every constant (H, xi, K, kappa, theta, paths)
├── ONZE_spec.md             # technical spec — the intent / source of truth for design
├── requirements.txt         # pinned Python dependencies
├── Makefile                 # install / ingest / ratings / fit / backtest / api / test / frontend / all
├── data/
│   ├── raw/                 # downloaded source CSVs (immutable)
│   └── processed/           # cleaned Parquet + onze.duckdb + reports
├── src/
│   ├── config.py            # loads config.yaml into a typed, frozen Config
│   ├── ingest/              # loaders, canonical team-name harmonisation, DuckDB schema, odds client
│   ├── ratings/elo.py       # Elo walk-forward, ratings_history, bare-Elo 1N2 baseline
│   ├── model/               # lambdas.py (Elo→λ), dixon_coles.py (P(x,y) + ρ MLE), knockout.py
│   ├── predict/             # fixture.py (pipeline), markets.py (derived), bracket.py (Monte-Carlo)
│   ├── eval/                # metrics.py, calibration.py, backtest.py (walk-forward)
│   └── api/main.py          # FastAPI app
├── frontend/                # React + Vite interface (see frontend/README.md)
├── docs/                    # METHODOLOGY.md, READING_OUTPUTS.md
├── reports/                 # generated backtest report + calibration artefacts
└── tests/                   # pytest suite (invariants, no-leakage, metrics)
```

### API endpoints (spec §4.4)

| Endpoint                     | Returns                                                        |
| ---------------------------- | ------------------------------------------------------------- |
| `GET  /fixtures`             | remaining matches + metadata                                  |
| `GET  /predict/{fixture_id}` | the `P(x, y)` matrix + all derived markets                    |
| `GET  /bracket`              | per-team reach/champion probabilities (Monte-Carlo)           |
| `GET  /eval/backtest?from=…` | model vs Elo (vs bookmaker) RPS + calibration                 |
| `POST /refresh`              | recompute predictions on new results/odds (the only heavy path) |

---

## Configuration

Every model constant lives in [`config.yaml`](config.yaml) — nothing is
hard-coded in the modules. The main knobs:

| Key               | Meaning                                                   | Default            |
| ----------------- | --------------------------------------------------------- | ------------------ |
| `H`               | Elo home-field bonus (points); 0 on neutral ground        | 100                |
| `base_rating`     | starting Elo for a team with no history                   | 1500               |
| `k_by_tournament` | Elo importance weight `K` per tournament type             | 10 / 25 / 60 / 20  |
| `xi`              | time-decay rate per day for the goal-model likelihood     | ln(2)/730 (~2-year half-life) |
| `K_max`           | score-matrix truncation (`x, y ∈ {0..K_max}`)             | 10                 |
| `kappa`           | extra-time prudence factor on λ                           | 0.8                |
| `theta`           | penalty-shootout sensitivity to Elo diff (0 = fair coin)  | 0.0                |

`xi` is not left at its default in the backtest: it is **tuned by out-of-sample
backtest RPS** over a documented grid of half-lives (never by eye).

---

## Status notes

- The seeded `fixtures` table currently holds **8 illustrative 2026 knockout
  fixtures** (a QF→SF→Final tree plus a third-place play-off) with canonical
  team ids — placeholders to be replaced by the real draw before production.
- The `odds` table is empty, so the bookmaker RPS baseline is guarded and
  reported as "pending odds ingestion". The de-vig code (normalisation + Shin
  refinement) and the comparison harness are already in place.
```
