import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useBacktest } from "../api/queries";
import { ServiceUnavailableError } from "../api/types";
import { ErrorState, Loading } from "../components/States";

// The spec's own methodological-honesty caveat (§3.7). The API does not
// currently serve a caveat string, so we surface the spec text. If a future API
// version adds one, prefer that (see NOTE below).
const HONESTY_CAVEAT =
  "Methodological honesty: the ~30 remaining live matches are a tiny sample. " +
  "The live tournament is a demo, not statistical validation — validation comes " +
  "from the historical walk-forward backtest. Lower RPS is better; the " +
  "de-vigged bookmaker line is the bar to beat, and getting close is already a " +
  "strong result.";

export function CalibrationView() {
  const { data, isLoading, error, refetch, isFetching } = useBacktest();

  // Distinguish "not built yet" (503) from a real failure.
  if (error instanceof ServiceUnavailableError) {
    return <BacktestPending detail={error.message} onRetry={() => refetch()} isRetrying={isFetching} />;
  }
  if (isLoading) return <Loading label="Loading backtest…" />;
  if (error) return <ErrorState error={error} />;
  if (!data) return <BacktestPending detail="No report returned." onRetry={() => refetch()} isRetrying={isFetching} />;

  return <CalibrationDashboard />;
}

function CalibrationDashboard() {
  const { data } = useBacktest();

  const rpsData = useMemo(() => {
    if (!data) return [];
    return data.results
      .filter((r) => r.rps != null)
      .map((r) => ({ name: r.name, rps: r.rps as number, n: r.n_matches }));
  }, [data]);

  if (!data) return null;

  // Highlight the model vs Elo comparison; colour the best (lowest) RPS.
  const bestRps = rpsData.length
    ? Math.min(...rpsData.map((r) => r.rps))
    : 0;

  return (
    <div>
      <h1>Calibration &amp; scoring</h1>
      <p className="subtitle">
        Backtest report{data.from_year ? ` from ${data.from_year}` : ""}. Source:{" "}
        <span className="mono">{data.source}</span>.
      </p>

      <div className="caveat" role="note">
        {HONESTY_CAVEAT}
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>RPS: model vs baselines (lower is better)</h2>
          {rpsData.length ? (
            <ResponsiveContainer width="100%" height={Math.max(200, rpsData.length * 46)}>
              <BarChart
                data={rpsData}
                layout="vertical"
                margin={{ left: 20, right: 50, top: 4, bottom: 4 }}
              >
                <CartesianGrid stroke="#2a3350" horizontal={false} />
                <XAxis type="number" stroke="#93a0bd" fontSize={11} />
                <YAxis type="category" dataKey="name" width={130} stroke="#93a0bd" fontSize={12} />
                <Tooltip
                  formatter={(v: number) => [v.toFixed(4), "RPS"]}
                  contentStyle={{
                    background: "#141a2e",
                    border: "1px solid #2a3350",
                    borderRadius: 8,
                    color: "#dfe6f2",
                  }}
                />
                <Bar dataKey="rps" radius={[0, 4, 4, 0]}>
                  {rpsData.map((r, i) => (
                    <Cell key={i} fill={r.rps === bestRps ? "#4fd18b" : "#5aa9ff"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="dim">The report contained no RPS figures.</p>
          )}
        </div>

        <div className="card">
          <h2>All reported metrics</h2>
          <div style={{ overflowX: "auto" }}>
            <table className="fixtures">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="right">RPS</th>
                  <th className="right">Log-loss</th>
                  <th className="right">Brier</th>
                  <th className="right">n</th>
                </tr>
              </thead>
              <tbody>
                {data.results.map((r) => (
                  <tr key={r.name}>
                    <td>{r.name}</td>
                    <td className="right mono">{r.rps != null ? r.rps.toFixed(4) : "—"}</td>
                    <td className="right mono">{r.log_loss != null ? r.log_loss.toFixed(4) : "—"}</td>
                    <td className="right mono">{r.brier != null ? r.brier.toFixed(4) : "—"}</td>
                    <td className="right mono">{r.n_matches ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h2>Reliability curve (predicted vs observed)</h2>
        {/* The current /eval/backtest response (src/api/main.py BacktestOut)
            returns only per-model scores, not per-bin calibration data. Rather
            than invent points, we flag this for the coach and reserve the slot.
            When the API adds a `calibration` array of {p_pred, p_obs, n}, plot
            it here against the y = x diagonal. */}
        <div className="slot-empty">
          The API&apos;s backtest report does not yet include per-bin calibration
          data ({"{p_pred, p_obs, n}"}), so the reliability curve against the
          diagonal cannot be drawn. Flagged for the coach: expose calibration
          bins from <span className="mono">src/eval/calibration.py</span> in the
          backtest report to enable this chart.
        </div>
      </div>
    </div>
  );
}

function BacktestPending({
  detail,
  onRetry,
  isRetrying,
}: {
  detail: string;
  onRetry: () => void;
  isRetrying: boolean;
}) {
  return (
    <div>
      <h1>Calibration &amp; scoring</h1>
      <div className="card">
        <h2>Backtest pending</h2>
        <p className="dim">
          The evaluation module has not produced a backtest report yet. This view
          will populate once the walk-forward backtest has run.
        </p>
        <p className="dim mono" style={{ fontSize: 12 }}>
          API said: {detail}
        </p>
        <button className="btn" onClick={onRetry} disabled={isRetrying}>
          {isRetrying ? "Checking…" : "Check again"}
        </button>
      </div>
      <div className="caveat" role="note" style={{ marginTop: 18 }}>
        {HONESTY_CAVEAT}
      </div>
    </div>
  );
}
