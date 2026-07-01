# ONZE frontend

React + Vite + TypeScript interface for ONZE — the World Cup 2026 score
distribution engine (spec §5). It consumes the FastAPI backend (`src/api/main.py`)
read-only and presents probabilities **honestly**: everything is a distribution,
nothing is framed as a sure thing.

## Views

1. **Fixtures** (`/`) — one row per remaining match: most-likely score, a stacked
   1N2 bar, qualification % for knockout ties, and a hedged confidence cue.
   Sortable by date or stage; click a row for the match detail.
2. **Match detail** (`/match/:fixtureId`) — the centrepiece: a heatmap of the
   `P(x, y)` score matrix with **home goals on the vertical axis and away goals on
   the horizontal axis, both labelled by team**. Plus a derived-markets panel
   (result 1N2, over/under lines, BTTS, advance % for KO) and the top-5 scores. A
   reserved slot overlays the bookmaker's implied probability once the API serves
   odds.
3. **Bracket** (`/bracket`) — per-team Monte-Carlo advancement odds
   (reach semi / final / champion). Updates when data refetches.
4. **Calibration** (`/calibration`) — model-vs-baseline RPS / log-loss / Brier
   from the backtest, plus the methodological-honesty caveat. Shows a friendly
   "backtest pending" state when the eval endpoint returns 503.

## Requirements

- Node 22+ and npm (Node 22 is used by the Dockerfile).

## Getting started

```bash
npm install      # install dependencies
npm run dev      # start the Vite dev server on http://localhost:5173
npm run build    # type-check (tsc -b) + production build into dist/
npm run preview  # serve the production build locally
```

## Configuration

The API base URL is configurable via a Vite env var:

| Variable             | Default                 | Description                     |
| -------------------- | ----------------------- | ------------------------------- |
| `VITE_API_BASE_URL`  | `http://localhost:8000` | Base URL of the ONZE FastAPI backend |

Set it in a `.env` file (see `.env.example`):

```bash
cp .env.example .env
# edit VITE_API_BASE_URL if the backend is elsewhere
```

CORS for the Vite dev origin is already enabled on the API
(`ONZE_CORS_ORIGINS`, defaults include `http://localhost:5173`).

## Data fetching

All server state goes through **React-Query** (`src/api/queries.ts`); the HTTP
client and typed response models live in `src/api/`. A 503 from
`/eval/backtest` is surfaced as a typed `ServiceUnavailableError` so the
calibration view can render its "pending" placeholder instead of failing. The
backend does **not** need to be running to build — fetches happen at runtime.

## Honesty notes for the coach

- The `/eval/backtest` response (`BacktestOut` in `src/api/main.py`) currently
  returns only per-model scores — no per-bin calibration data and no caveat
  string. The reliability-curve slot is therefore reserved and flagged in-app;
  expose calibration bins (`{p_pred, p_obs, n}`) from
  `src/eval/calibration.py` to enable that chart. The honesty caveat shown is the
  spec's own §3.7 text.
- `/bracket` returns per-team reach/champion probabilities, not a fixed tie tree,
  so progression is shown as columns rather than a drawn bracket.
