import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useBacktest } from "../api/queries";
import { ServiceUnavailableError } from "../api/types";
import type { Backtest } from "../api/types";
import { ErrorState, Loading } from "../components/States";

// The spec's own methodological-honesty caveat (§3.7), used as a fallback when
// the API does not serve its own `caveat` string. Prefer the API's when present.
const FALLBACK_CAVEAT =
  "Methodological honesty: the ~30 remaining live matches are a tiny sample. " +
  "The live tournament is a demo, not statistical validation — validation comes " +
  "from the historical walk-forward backtest. Lower RPS is better; the " +
  "de-vigged bookmaker line is the bar to beat, and getting close is already a " +
  "strong result.";

export function CalibrationView() {
  const { data, isLoading, error, refetch, isFetching } = useBacktest();

  // Distinguish "not built yet" (503) from a real failure.
  if (error instanceof ServiceUnavailableError) {
    return (
      <BacktestPending detail={error.message} onRetry={() => refetch()} isRetrying={isFetching} />
    );
  }
  if (isLoading) return <Loading label="Loading backtest…" />;
  if (error) return <ErrorState error={error} />;
  if (!data)
    return (
      <BacktestPending
        detail="No report returned."
        onRetry={() => refetch()}
        isRetrying={isFetching}
      />
    );

  return <CalibrationDashboard data={data} />;
}

function CalibrationDashboard({ data }: { data: Backtest }) {
  const rpsData = useMemo(
    () =>
      data.results
        .filter((r) => r.rps != null)
        .map((r) => ({ name: r.name, rps: r.rps as number, n: r.n_matches })),
    [data],
  );

  const caveat = data.caveat && data.caveat.trim() ? data.caveat : FALLBACK_CAVEAT;
  const bestRps = rpsData.length ? Math.min(...rpsData.map((r) => r.rps)) : 0;
  const hasCalibration = Array.isArray(data.calibration) && data.calibration.length > 0;

  return (
    <div>
      <h1>Calibration &amp; scoring</h1>
      <p className="subtitle">
        Backtest report{data.from_year ? ` from ${data.from_year}` : ""}. Source:{" "}
        <span className="mono">{data.source}</span>.
      </p>

      {(data.ece != null || data.xi_half_life_days != null || data.generated_from != null) && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 16,
            alignItems: "baseline",
            marginBottom: 16,
          }}
        >
          {data.ece != null && (
            <div>
              <div className="dim" style={{ fontSize: 12 }}>
                Expected Calibration Error
              </div>
              <div style={{ fontSize: 26, fontWeight: 700 }} className="mono">
                {data.ece.toFixed(3)}
              </div>
            </div>
          )}
          <div className="dim" style={{ fontSize: 12 }}>
            {data.xi_half_life_days != null &&
              `tuned ξ half-life ${Math.round(data.xi_half_life_days)}d`}
            {data.xi_chosen != null && ` (ξ = ${data.xi_chosen.toPrecision(3)})`}
            {data.generated_from && ` · window from ${data.generated_from}`}
            {data.n_matches != null && ` · ${data.n_matches.toLocaleString()} matches`}
          </div>
        </div>
      )}

      <div className="caveat" role="note">
        {caveat}
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
                  contentStyle={tooltipStyle}
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
          <h2>Reliability curve (predicted vs observed)</h2>
          {hasCalibration ? (
            <ReliabilityCurve data={data} />
          ) : (
            <div className="slot-empty">
              Reliability curve pending: this backtest artifact did not include
              per-bin calibration data ({"{p_pred, p_obs, n}"}). It will render
              automatically once the eval report provides it.
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
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
  );
}

const tooltipStyle = {
  background: "#141a2e",
  border: "1px solid #2a3350",
  borderRadius: 8,
  color: "#dfe6f2",
} as const;

function ReliabilityCurve({ data }: { data: Backtest }) {
  const bins = data.calibration ?? [];

  // Points: predicted probability (x) vs observed frequency (y), sized by n.
  const points = bins
    .filter((b) => b.n > 0)
    .map((b) => ({ p_pred: b.p_pred, p_obs: b.p_obs, n: b.n }));

  // Perfect-calibration diagonal (y = x) as a reference line.
  const diagonal = [
    { p_pred: 0, p_obs: 0, n: 0 },
    { p_pred: 1, p_obs: 1, n: 0 },
  ];

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 24 }}>
          <CartesianGrid stroke="#2a3350" />
          <XAxis
            type="number"
            dataKey="p_pred"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke="#93a0bd"
            fontSize={11}
            label={{
              value: "Predicted probability",
              position: "insideBottom",
              offset: -12,
              fill: "#93a0bd",
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            dataKey="p_obs"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke="#93a0bd"
            fontSize={11}
            label={{
              value: "Observed frequency",
              angle: -90,
              position: "insideLeft",
              fill: "#93a0bd",
              fontSize: 12,
            }}
          />
          <ZAxis type="number" dataKey="n" range={[40, 400]} name="n" />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={tooltipStyle}
            formatter={(value: number, name: string) => {
              if (name === "n") return [value.toLocaleString(), "samples (n)"];
              if (name === "p_obs") return [value.toFixed(3), "observed"];
              if (name === "p_pred") return [value.toFixed(3), "predicted"];
              return [value, name];
            }}
          />
          {/* y = x perfect-calibration reference (rendered as an invisible-point
              line series so only the dashed line shows). */}
          <Scatter
            data={diagonal}
            line={{ stroke: "#8b93a7", strokeDasharray: "5 5" }}
            shape={() => <g />}
            legendType="none"
            isAnimationActive={false}
          />
          {/* Actual bins: point per bin, connected, area sized by n. */}
          <Scatter
            data={points}
            fill="#5aa9ff"
            line={{ stroke: "#5aa9ff", strokeWidth: 1 }}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="dim" style={{ fontSize: 12, marginTop: 8 }}>
        Dashed line = perfect calibration (y = x). Points on it are well
        calibrated; above means the outcome happened more often than predicted,
        below means less. Point area is proportional to the number of predictions
        in each bin.
      </p>
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
        {FALLBACK_CAVEAT}
      </div>
    </div>
  );
}
