import { useMemo } from "react";
import { heatColor, heatTextColor } from "../lib/colorScale";
import { pct1 } from "../lib/format";

/**
 * Heatmap of the joint score matrix P(x, y).
 *
 * API contract (src/api/main.py): matrix[x][y] = P(home scores x, away scores y).
 * So the ROW index is HOME goals and the COLUMN index is AWAY goals. We render
 * HOME goals down the vertical axis and AWAY goals across the horizontal axis,
 * with both axes clearly labelled by team name.
 */
export function ScoreHeatmap({
  matrix,
  homeTeam,
  awayTeam,
  maxGoals = 6,
}: {
  matrix: number[][];
  homeTeam: string;
  awayTeam: string;
  /** Truncate display to keep cells legible; tail mass is negligible. */
  maxGoals?: number;
}) {
  const { rows, cols, peakMax } = useMemo(() => {
    const r = Math.min(maxGoals + 1, matrix.length);
    const c = Math.min(maxGoals + 1, matrix[0]?.length ?? 0);
    let mx = 0;
    for (let x = 0; x < r; x++)
      for (let y = 0; y < c; y++) mx = Math.max(mx, matrix[x]?.[y] ?? 0);
    return { rows: r, cols: c, peakMax: mx || 1 };
  }, [matrix, maxGoals]);

  return (
    <div>
      <div className="heatmap-wrap">
        <div className="heat-axis-away">{awayTeam} goals →</div>
        <div className="heat-axis-home">{homeTeam} goals →</div>
        <div className="heat-body">
          <div
            className="heat-grid"
            style={{
              gridTemplateColumns: `auto repeat(${cols}, 1fr)`,
            }}
            role="table"
            aria-label={`Score probability matrix: ${homeTeam} (rows) vs ${awayTeam} (columns)`}
          >
            {/* header row: away goal counts */}
            <div className="heat-colhdr" aria-hidden />
            {Array.from({ length: cols }, (_, y) => (
              <div key={`ch-${y}`} className="heat-colhdr">
                {y}
              </div>
            ))}

            {Array.from({ length: rows }, (_, x) => (
              <Row
                key={`r-${x}`}
                x={x}
                cols={cols}
                matrix={matrix}
                peakMax={peakMax}
                homeTeam={homeTeam}
                awayTeam={awayTeam}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="legend-scale">
        <span>less likely</span>
        <div
          className="legend-gradient"
          style={{
            background: `linear-gradient(90deg, ${heatColor(0)}, ${heatColor(
              0.3,
            )}, ${heatColor(0.6)}, ${heatColor(0.85)}, ${heatColor(1)})`,
          }}
        />
        <span>more likely</span>
        <span className="dim" style={{ marginLeft: "auto" }}>
          White outline = most likely score. Values are probabilities, not
          predictions.
        </span>
      </div>
    </div>
  );
}

function Row({
  x,
  cols,
  matrix,
  peakMax,
  homeTeam,
  awayTeam,
}: {
  x: number;
  cols: number;
  matrix: number[][];
  peakMax: number;
  homeTeam: string;
  awayTeam: string;
}) {
  return (
    <>
      <div className="heat-rowhdr" style={{ display: "flex", alignItems: "center" }}>
        {x}
      </div>
      {Array.from({ length: cols }, (_, y) => {
        const p = matrix[x]?.[y] ?? 0;
        const t = p / peakMax;
        const isPeak = p === peakMax && p > 0;
        return (
          <div
            key={`c-${x}-${y}`}
            className={`heat-cell${isPeak ? " peak" : ""}`}
            style={{ background: heatColor(t), color: heatTextColor(t) }}
            title={`${homeTeam} ${x} – ${y} ${awayTeam}: ${pct1(p)}`}
          >
            {p >= 0.01 ? Math.round(p * 100) : ""}
          </div>
        );
      })}
    </>
  );
}
