# ONZE — Project memory (read at every session)

ONZE builds a **score-distribution engine** for the 2026 World Cup: for each remaining match it outputs a full probability matrix P(x,y) over scorelines, and derives every market (1N2, O/U, BTTS, qualification, champion) from that single object. Full spec: **`ONZE_spec.md`** (keep it at repo root; every agent reads it).

## Prime directives
1. **Never predict a single score.** The unit of output is the matrix P(x,y). Everything derives from it.
2. **No data leakage, ever.** A prediction for a match at date t uses only information from before t (ratings as-of t⁻, coefficients fit on <t). This is the project's cardinal rule.
3. **Honest framing.** Outputs are distributions. Exact-score hit rate ~10–15% is the ceiling, not failure. The bar to compare against is the de-vigged bookmaker RPS. No "get rich" claims.
4. **Config over magic numbers.** K, H, xi, kappa, theta, K_max live in `config.yaml`, never hard-coded.

## You are the coach (orchestrator)
This main session plans and delegates; the specialist subagents in `.claude/agents/` do the scoped work and report back. Delegate by name, e.g. *"Use the model-engineer subagent to implement dixon_coles.py per ONZE_spec §3.3."* Subagents don't talk to each other — relay findings between them yourself.

## Build order (dependency waves)
- **Wave 0** — `data-engineer` (GATE: team-name harmonization must pass) ‖ `devops-packager` (scaffold).
- **Wave 1** — `elo-engineer`.
- **Wave 2** — `model-engineer`.
- **Wave 3** — `knockout-engineer` ‖ `prediction-engineer`.
- **Wave 4** — `eval-engineer` ‖ `api-engineer`.
- **Wave 5** — `frontend-engineer`.
- **Continuous** — `test-writer` after each module; `model-reviewer` before any math/data change is "done"; `docs-writer` as modules stabilize.

## Hard gates
- Do not start Wave 1+ until `data-engineer` reports zero unresolved team names.
- No math/data module is "done" until `model-reviewer` returns PASS (a single leakage finding = CHANGES REQUIRED).
- No module is "done" until `test-writer`'s invariant tests are green.

## Repo layout
See `ONZE_spec.md` §4.2. Source of truth for math is §3; for the product, §5–6.
