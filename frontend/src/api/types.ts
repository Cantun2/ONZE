// TypeScript types mirroring the ONZE FastAPI Pydantic response models
// (src/api/main.py). Field names match the API exactly — do not rename.

export interface Fixture {
  fixture_id: number;
  date: string | null;
  home_id: number;
  away_id: number;
  home_team: string;
  away_team: string;
  stage: string | null;
  neutral: boolean;
  host_flag: boolean;
  knockout: boolean;
}

export interface ScoreProb {
  home: number;
  away: number;
  prob: number;
}

export interface OverUnderLine {
  over: number;
  under: number;
}

export interface AdvanceProbs {
  p_home_advance: number;
  p_away_advance: number;
}

export interface Prediction {
  fixture_id: number;
  model_version: string;
  home_team: string;
  away_team: string;
  stage: string | null;
  knockout: boolean;
  // matrix[x][y] = P(home scores x, away scores y).
  matrix: number[][];
  p_home: number;
  p_draw: number;
  p_away: number;
  most_likely_score: string;
  top5_scores: ScoreProb[];
  // Keyed by line, e.g. "0.5", "1.5", ..., "4.5".
  over_under: Record<string, OverUnderLine>;
  btts: number;
  advance: AdvanceProbs | null;
}

export interface BracketTeam {
  team_id: number;
  team: string;
  p_reach_semi: number;
  p_reach_final: number;
  p_champion: number;
}

export interface Bracket {
  n_sims: number;
  seed: number;
  teams: BracketTeam[];
}

export interface BacktestRow {
  name: string;
  rps: number | null;
  log_loss: number | null;
  brier: number | null;
  n_matches: number | null;
}

export interface Backtest {
  from_year: number | null;
  results: BacktestRow[];
  source: string;
}

// Thrown for a 503 so views can render a friendly "pending" placeholder.
export class ServiceUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ServiceUnavailableError";
  }
}
