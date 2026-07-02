# How to read ONZE's outputs

A one-page guide to reading the interface honestly. ONZE outputs
**distributions**, not tips. Here is how to interpret each thing you see.

---

## The heatmap is the whole prediction

The centrepiece of a match view is the **score-matrix heatmap** — the object
`P(x, y) = P(home scores x, away scores y)`. In the UI, **home goals run down the
vertical axis, away goals across the horizontal axis**, and colour intensity is
the probability of that exact scoreline.

This grid **is** the prediction. Everything else on the page (1N2 bar, over/under,
BTTS, most-likely score, qualification %) is just a *sum over cells of this
grid*. That is deliberate: because the markets are re-summed from the one matrix,
they can never contradict each other.

**How to read it:** look at where the mass concentrates, not at a single cell. A
diffuse, spread-out heatmap means an uncertain, open game; a tight cluster near
the low-score corner means a likely cagey, low-scoring match. The three triangular
regions tell you the result:

- below the diagonal (`x > y`) → **home win**
- on the diagonal (`x = y`) → **draw**
- above the diagonal (`x < y`) → **away win**

---

## The "most-likely score" is just the mode — and it is usually unlikely

The most-likely score is the single brightest cell (`argmax P(x, y)`). It is the
*mode* of the distribution, nothing more. **Its probability is typically well
under 20%** — often 8–12%. A most-likely score of "1-0" at 11% means "1-0 is the
single most probable outcome, and it will still fail to happen ~89% of the time."

Do **not** read the most-likely score as a prediction of the final result. Read
the *distribution*.

---

## Exact-score hit rate: ~10–15% is the ceiling, not a failure

Football scorelines are intrinsically hard to call exactly. Even a perfect model
cannot exceed roughly a **10–15% exact-score hit rate**, because that much
randomness is baked into the sport. So:

- A ~10–15% exact-score hit rate is **success**, not a shortfall.
- ONZE never reports a single exact-score accuracy number as "the score". It is a
  poor, misleading metric for a distributional model, and we don't optimise for
  it.

If you want to judge whether ONZE is any good, look at the two things below.

---

## The bar to beat: de-vigged bookmaker RPS

The honest way to judge the model is **RPS** (Ranked Probability Score) on the
1N2 outcome — lower is better. And the meaningful bar is not zero, it is the
**de-vigged bookmaker's RPS**: the closing odds with the margin removed. Books
aggregate enormous information; getting *close* to their RPS is already an
excellent result.

- **Beating bare Elo** (the model-free baseline) shows the goal model adds value
  over the rating alone. ONZE does this (~0.168 vs ~0.170 on the backtest).
- **Matching the de-vigged bookmaker** is the real target. In the current build
  this comparison is **pending odds ingestion** — the harness is in place and
  lights up automatically once closing odds are loaded.

One number in isolation means little; RPS is only interpretable *relative to*
these baselines.

---

## Calibration is how you judge honesty

A model can look sharp and still lie about its confidence. **Calibration** asks:
when ONZE says "70%", does that thing happen ~70% of the time?

The **reliability diagram** (Calibration view) plots predicted probability
against observed frequency. A well-calibrated model sits on the diagonal. The
summary number is the **Expected Calibration Error (ECE)** — the average gap
between "said" and "happened". **Lower is better; ~0.01–0.03 is well-calibrated.**
ONZE's backtest lands around **ECE ≈ 0.017**.

If the model is well-calibrated, its probabilities mean what they say — which is
the whole point of a distribution engine.

---

## What the tournament / bracket numbers are (and aren't)

The bracket view shows each team's `P(reach semi / final)` and `P(champion)` from
a Monte-Carlo simulation of the remaining tree. These are consistent with the
per-match qualification numbers and reproducible (fixed seed). Champion
probabilities across all teams sum to 1.

Treat the live 2026 tournament as a **demo**, not a scoreboard for the model. The
~30 remaining matches are far too few to validate anything — the real validation
is the multi-decade historical backtest. Enjoy the tournament view for what it
is: a fun, honest read of the odds, not a verdict on the model.

---

## TL;DR

| You see…                     | Read it as…                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| Heatmap                      | the full distribution — *the* prediction; look at the mass         |
| Most-likely score            | just the mode; usually <20% likely; not "the result"               |
| Exact-score hit rate         | ~10–15% is the ceiling; success, not failure                       |
| 1N2 / O/U / BTTS             | sums over the same matrix; mutually consistent by construction     |
| RPS                          | the score to judge on — vs Elo, and vs the de-vigged book (the bar) |
| Calibration / ECE            | whether the probabilities are honest (~0.01–0.03 = good)           |
| Bracket / champion %         | a reproducible demo, not a validation of the model                 |
