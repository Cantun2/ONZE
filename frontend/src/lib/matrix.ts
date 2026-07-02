// Derivations from the joint score matrix P(x, y) returned by GET /predict/{id}.
// matrix[x][y] = P(home scores x, away scores y).

export interface ExpectedGoals {
  xgHome: number;
  xgAway: number;
}

/**
 * Expected (model) goals for each side, derived from the score matrix:
 *   xg_home = Σ_x x · (Σ_y P[x][y])
 *   xg_away = Σ_y y · (Σ_x P[x][y])
 * These are the model's mean goals, NOT a prediction of the actual scoreline.
 */
export function expectedGoals(matrix: number[][]): ExpectedGoals {
  let xgHome = 0;
  let xgAway = 0;
  for (let x = 0; x < matrix.length; x++) {
    const row = matrix[x];
    if (!row) continue;
    for (let y = 0; y < row.length; y++) {
      const p = row[y];
      xgHome += x * p;
      xgAway += y * p;
    }
  }
  return { xgHome, xgAway };
}

/** Round a goal expectation to one decimal for display (unit-safe). */
export function fmtGoals(x: number): string {
  if (x == null || Number.isNaN(x)) return "–";
  return x.toFixed(1);
}
