# ONZE — Methodology

This document translates the mathematics of ONZE for a reader, and ties every
formula to the **implemented code** (file paths and function names). Where the
code and the spec (`ONZE_spec.md`) diverge, this document follows the code and
notes the divergence.

The whole engine rests on one idea: never predict a single scoreline. The unit
of output is the joint score distribution **P(X = x, Y = y)** (home goals `x`,
away goals `y`), and every market is a deterministic sum over its cells.

Contents:

1. [Elo ratings](#31-elo-ratings) — `src/ratings/elo.py`
2. [Elo → expected goals λ](#32-elo--expected-goals-λ) — `src/model/lambdas.py`
3. [Dixon-Coles goals model](#33-dixon-coles-goals-model) — `src/model/dixon_coles.py`
4. [Derived markets](#34-derived-markets) — `src/predict/markets.py`
5. [Knockout resolution](#35-knockout-resolution) — `src/model/knockout.py`
6. [No-leakage walk-forward](#36-no-leakage-walk-forward)
7. [Evaluation](#37-evaluation--calibration--baselines) — `src/eval/`

---

## 3.1 Elo ratings

**File:** `src/ratings/elo.py`

Each team carries a single rating `R`. Ratings are updated one match at a time,
in strict chronological order. The *expected* score of the home team is the
400-point logistic (`expected_home_score`):

```
We = 1 / (1 + 10 ** (-(R_home + H_eff - R_away) / 400))
```

- `H_eff` is the **effective home bonus**: `H` (config, default 100) for a
  genuine host, and `0` on neutral ground. The rule lives in `_home_bonus`:
  a `host_flag` team always gets `H` (so 2026 hosts USA/Canada/Mexico keep their
  advantage even at a "neutral" venue); otherwise a `neutral` match gives 0.

The actual result `W ∈ {1, 0.5, 0}` (win/draw/loss, from `_result_from_goals`),
and the update is scaled by a **goal-difference multiplier** `G`
(`goal_diff_multiplier`):

```
G = 1              if |Δ| ≤ 1
    1.5            if |Δ| = 2
    (11 + |Δ|) / 8 if |Δ| ≥ 3
```

The zero-sum update (total rating conserved) is:

```
R'_home = R_home + K · G · (W − We)
R'_away = R_away − K · G · (W − We)
```

`K` is the match importance, stored per row in `matches.importance_k` (friendly
≈ 10, qualifier ≈ 25, World Cup finals ≈ 60; config `k_by_tournament`). A row's
own `importance_k` is authoritative; the config map is only a fallback.

**Walk-forward, leak-free (`compute_elo_history`).** For each match the function
first *records* both teams' current (pre-match) ratings into `ratings_history`,
*then* applies that match's result. So a stored rating is always a function of
strictly-earlier matches only — this is what makes the backtest of §3.6 valid.
Unplayed / null-score rows are skipped for the update and produce no history row.
On the real data this produces **98,968 history rows** from **49,496 matches**.

The pre-match lookup for downstream code is `rating_as_of(team_id, date, …)`,
which returns a team's last stored rating **strictly before** `date`.

**Bare-Elo 1N2 baseline (`elo_1x2`).** The honest model-free baseline the goal
model must beat. It turns `We` into win/draw/loss with an explicit draw
allocation `p_draw = draw_factor · (1 − |2·We − 1|)` (default `draw_factor` 0.30,
peaking for even matches), then splits the rest around `We` so that
`We = p_home + 0.5·p_draw`, clamps and renormalises.

---

## 3.2 Elo → expected goals λ

**File:** `src/model/lambdas.py`

The mean of the goals model is a **global-coefficient Poisson regression** with a
log link (`predict_lambdas`):

```
log λ_home = c0 + c1 · (R_home + H_eff − R_away)
log λ_away = c0 + c1 · (R_away − R_home − H_eff)
```

The differential is *anti-symmetric*: the away side sees the negation of the home
side's differential. Only two parameters (`c0`, `c1`) are fit here — plus `ρ` in
Dixon-Coles — over tens of thousands of matches, so the estimate is very stable.
`c0` sets the average scoring level; `c1 > 0` the sensitivity to strength.

**Leak-free feature build (`match_features` / `build_design`).** For every played
match the code attaches each team's *pre-match* Elo. It does this with a
**row-exact `(team_id, appearance-sequence)` join** (`_prematch_elos_by_sequence`)
against `ratings_history`, *not* a calendar-date `merge_asof`. Since the Elo
engine emits its two pre-match rows per match in `(date, match_id)` order
(home then away), a team's *k*-th appearance lines up with its *k*-th history
row. This is immune to same-day double-headers (a team playing twice on one day
gets each match's own pre-match rating, never its sibling's post-result rating).

**Fit (`fit_lambda_coeffs`).** A standard Poisson GLM: both sides of each match
are stacked into one design (`[1, diff]` → `goals`) and the Poisson
log-likelihood is maximised. Implementation uses `statsmodels` GLM (IRLS) when
available, falling back to a hand-rolled scipy L-BFGS-B Newton fit (`_fit_scipy`)
— both maximise the same likelihood.

Fitted values on the full history: **c0 ≈ 0.284, c1 ≈ 0.00199**.

> **Note on the spec.** The spec (§3.2) writes the joint fit as maximising a
> single Poisson likelihood over `(c0, c1)` and estimating `ρ` jointly. The code
> fits `(c0, c1)` by GLM first, then `ρ` by a 1-D time-weighted MLE
> (see `_fit_coeffs_and_rho`). This is a documented sequencing choice: the
> objective is dominated by the Poisson means and `ρ` is a small low-score
> refinement, so the two-stage fit gives essentially the joint optimum.

---

## 3.3 Dixon-Coles goals model

**File:** `src/model/dixon_coles.py`

**Naïve independent Poisson.** If `X ~ Poisson(λ)` and `Y ~ Poisson(μ)`
independently, `P(x, y) = Pois(x; λ) · Pois(y; μ)`. This under-weights low, tight
scores (0-0, 1-1) and ignores the correlation between the two scores.

**Dixon-Coles correction (`tau`).** Multiply the four low-score cells by `τ`:

```
τ(x, y) = 1 − λ·μ·ρ   if (x, y) = (0, 0)
          1 + λ·ρ      if (x, y) = (0, 1)
          1 + μ·ρ      if (x, y) = (1, 0)
          1 − ρ        if (x, y) = (1, 1)
          1            otherwise
```

The joint law (the boxed spec formula):

```
P(X = x, Y = y) = τ(x, y) · Pois(x; λ) · Pois(y; μ)
```

`ρ` is the dependence parameter. Validity requires `τ ≥ 0` on all four low
scores; the code bounds the optimisation to `RHO_BOUNDS = (-0.2, 0.0)`, the
spec's realistic region. Empirically `ρ` is small and negative (independent
Poisson under-weights draws), so the bound is not binding at the optimum.

**Score matrix (`score_matrix(lam, mu, rho, K)`).** Builds the `(K+1)×(K+1)`
matrix: `outer(Pois(λ), Pois(μ))` with the four τ cells applied, then
**truncated at `K` and renormalised** so it sums to 1 (spec §7.5). `K` defaults
to `config.K_max = 10`. It is numerically safe as `λ` or `μ → 0` (mass collapses
onto the 0-goal row/column, no NaN), and reduces to the independent-Poisson outer
exactly when `ρ = 0` (asserted in `_self_check_score_matrix`).

**Time decay.** Older matches count less. Each observation is weighted by

```
φ(Δt) = exp(−ξ · Δt),   Δt = (t_ref − date) in DAYS
```

where `t_ref` is the most recent match in the fit window (so the newest match has
weight 1). `ξ` is `config.xi` (default `ln(2)/730`, a ~2-year half-life); it is
**tuned by backtest RPS**, not by eye (§3.7).

**Time-weighted log-likelihood (`_weighted_loglik`), term-for-term as in the
spec** (factorial constants dropped, as they don't depend on the parameters):

```
ℓ(c0, c1, ρ) = Σ_m φ(Δt_m) · [ ln τ(x_m, y_m) − λ_m + x_m·ln λ_m − μ_m + y_m·ln μ_m ]
```

**Fit of ρ (`fit_rho`).** Given fixed `(c0, c1)`, maximise the time-weighted DC
log-likelihood over `ρ` with a bounded 1-D minimiser (`scipy.minimize_scalar`,
`method="bounded"`) on `RHO_BOUNDS`. The joint entry points are `fit_full_history`
(production coefficients on all played matches) and `fit_up_to(date)` (leak-free
window for the backtest — see §3.6). Fitted on the full history: **ρ ≈ −0.053**.

Coefficients are persisted to a `model_coeffs` table (`persist_coeffs`) keyed by
`model_version` (`"dc-v1"`) and read back by the predict / eval layers.

---

## 3.4 Derived markets

**File:** `src/predict/markets.py`

Every market is a **pure sum over the cells of `P`** (`derive_markets`), so the
markets can never disagree with the matrix. `P[x, y] = P(home scores x, away
scores y)`.

- **1N2 at 90'** (`one_x_two`) — the three regions partition the matrix:

  ```
  p_home = Σ_{x>y} P(x, y)   (strictly-lower triangle)
  p_draw = Σ_{x=y} P(x, y)   (main diagonal)
  p_away = Σ_{x<y} P(x, y)   (strictly-upper triangle)
  ```

- **Most-likely score** (`most_likely_score`) — `argmax_{(x,y)} P(x, y)`, the
  *mode*. **Top-5** (`top_n_scores`) — the five largest cells (ties broken by
  natural `(x, y)` order for reproducibility).

- **Over/Under line L** (`over_under`) — `over = Σ_{x+y > L} P(x, y)`,
  `under = 1 − over`. Lines are the half-integers 0.5 … 4.5 (`OVER_UNDER_LINES`)
  so no total lands exactly on a line.

- **BTTS (both teams score)** (`btts`) — by inclusion-exclusion:

  ```
  P(BTTS) = 1 − P(X = 0) − P(Y = 0) + P(0, 0)
  ```

---

## 3.5 Knockout resolution

**File:** `src/model/knockout.py`

A knockout tie has no draw; the winner is decided by chaining **three
strictly-conditional stages**, each entered only if the previous was level. The
module *consumes* the base score matrix — it never rewrites it.

**(a) Regulation, 90' (`resolve_stage`).** `P_90 = score_matrix(λ, μ, ρ, K)`,
split into `(home, away, level)` — home win `Σ_{x>y}`, away win `Σ_{x<y}`, level
`Σ_{x=y}`. If not level, the tie is resolved here.

**(b) Extra time (conditional on a 90' draw).** 30 minutes, played more
prudently, so the rates shrink:

```
λ_ET = λ · (30/90) · κ,   μ_ET = μ · (30/90) · κ
```

with `κ = config.kappa ≈ 0.8`. The *same* Dixon-Coles primitive is reused on this
mini-score: `P_ET = score_matrix(λ_ET, μ_ET, ρ, K)`, split into
`(home, away, level)`.

**(c) Penalties (conditional on still level after ET).** Historically ~50/50,
only weakly tied to strength (`penalty_prob`):

```
P_pen(home) = sigmoid(θ · (R_home − R_away))
```

`θ = config.theta`, default **0 → an exact fair coin** (0.5/0.5), independent of
Elo — the spec warns against over-interpreting shootouts. An optional
`calibrate_theta` can MLE-fit `θ` on the historical shootout record, but the
default path uses the config value.

**Total probability of advancing (`advance_prob`)**, by the law of total
probability:

```
P(home advances) = P_90(home) + P_90(draw) · [ P_ET(home) + P_ET(level) · P_pen(home) ]
```

and symmetrically for away. Because every stage is a proper partition and the
ET/penalty branches are entered *only* through the prior stage's level mass,
`p_home_advance + p_away_advance = 1` exactly (float tolerance) — no mass is lost
or double-counted (asserted in `_self_check`).

**Bracket simulation** (`src/predict/bracket.py`, `simulate_tournament`)
Monte-Carlos the remaining bracket tree: for each simulated tie it draws the
winner from a Bernoulli on `advance_prob(...)`, so the tournament outputs
(`P(reach semi/final)`, `P(champion)`) are exactly consistent with the per-fixture
qualification numbers. It is driven by a single seeded RNG (default seed
`20260619`), so results are bit-for-bit reproducible; champion probabilities sum
to 1.

---

## 3.6 No-leakage walk-forward

**Files:** `src/model/dixon_coles.py` (`fit_up_to`), `src/eval/backtest.py`

The cardinal rule: to predict a match at date `t`, use **only** information from
before `t` — ratings as-of `t⁻`, coefficients fit on matches `< t`. Any estimate
that touches a match on or after `t` invalidates the backtest.

The mechanics:

- **Coefficients:** `fit_up_to(block_start)` fits `(c0, c1, ρ)` on matches
  *strictly before* the block (`_played_before` splits on the date).
- **Ratings:** the as-of Elo lookup is strict-`<`. In the backtest the code uses
  `AsOfElo`, an in-memory per-team sorted-array + `bisect_left` mirror of
  `rating_as_of` (same strict-`<` semantics, verified against it in the tests),
  purely for speed over ~10⁵ lookups.
- **Assertions in code:** for every prediction the backtest asserts
  (1) `fit_window_max_date < block_start` and (2) `fit_window_max_date <
  match_date`. A leak raises immediately rather than silently inflating a score.

The walk-forward advances block by block (`walk_forward`): fit on the past,
predict the block, score, advance.

---

## 3.7 Evaluation, calibration & baselines

**Files:** `src/eval/metrics.py`, `src/eval/calibration.py`, `src/eval/backtest.py`

**RPS — Ranked Probability Score (`rps`).** The primary metric, for the *ordered*
1N2 outcome (home, draw, away). For `r = 3` categories:

```
RPS = (1 / (r − 1)) · Σ_{i=1}^{r−1} ( Σ_{j≤i} (p_j − e_j) )²
```

`p` = predicted probabilities, `e` = one-hot of the realised outcome, both in the
order (home, draw, away). Implemented via cumulative sums, dropping the final
(always ~0) cumulative term. RPS rewards putting mass *near* the true ordered
outcome. Lower is better; range `[0, 1]`.

**log-loss & Brier (`log_loss`, `brier`).** For the binary markets (BTTS,
Over/Under 2.5): `−[y·ln p + (1−y)·ln(1−p)]` and `(p − y)²`. Log-loss clamps `p`
to `[ε, 1−ε]` so a confident-but-wrong prediction is heavily but finitely
penalised.

**Calibration (`src/eval/calibration.py`).** A model can score well on RPS yet be
mis-calibrated (says "70%", happens 55% of the time). The 1N2 predictions are
flattened one-vs-rest (`onehot_flatten`: each match contributes three
`(prob, indicator)` points), binned into a **reliability table**
(`reliability_table`: mean predicted vs observed frequency per bin), and
summarised as the **Expected Calibration Error** (`expected_calibration_error`),
the count-weighted mean absolute gap. A perfectly calibrated model sits on the
diagonal. Rendering (`render`) writes a matplotlib PNG when available, else
degrades to CSV + markdown (headless-safe).

**Baselines (spec §3.7).**

1. **Bare Elo** (`elo_1x2`) — the model-free 1N2 bar. Beating it shows the goal
   model adds value over the rating alone.
2. **De-vigged bookmaker** — the *real* bar. Closing odds → implied probabilities
   with the vig removed, either by simple normalisation
   (`devig_normalisation`: `p_i = (1/o_i) / Σ_k (1/o_k)`) or Shin's method
   (`devig_shin`, a less-biased refinement). Matching the book's RPS is an
   excellent result. **This baseline is currently guarded**: the `odds` table is
   empty, so the backtest reports "pending odds ingestion" and the de-vig
   harness activates automatically once odds are loaded.

**ξ tuning (`tune_xi`).** The time-decay `ξ` is chosen by **out-of-sample
backtest RPS** over a documented grid of half-lives (`half_life_to_xi`), never by
eye. The chosen value and the full search are written into the report.

**Results (committed smoke window, 2024–2026, 2,623 matches):** model RPS
**0.16844** vs bare-Elo **0.17040** (model beats Elo); **ECE ≈ 0.017**. See
`reports/BACKTEST_REPORT.md`.

**Honesty caveat (spec §3.7 note).** The statistically meaningful validation is
the multi-decade walk-forward backtest on tens of thousands of matches. The ~30
remaining 2026 World Cup matches are a **demo, not a validation set** — 30
matches cannot separate models. ONZE deliberately reports RPS distributions,
calibration and multiple markets, and **never a single exact-score accuracy
number as "success"**: exact-score guessing is a poor, misleading metric for a
distributional model.
