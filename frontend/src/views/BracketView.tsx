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
import { useBracket } from "../api/queries";
import { ErrorState, Loading, EmptyState } from "../components/States";
import { pct, pct1 } from "../lib/format";

export function BracketView() {
  const { data, isLoading, error, isFetching } = useBracket();

  const teams = useMemo(() => {
    if (!data) return [];
    return [...data.teams].sort((a, b) => b.p_champion - a.p_champion);
  }, [data]);

  if (isLoading) return <Loading label="Simulating tournament…" />;
  if (error) return <ErrorState error={error} />;
  if (!data || teams.length === 0)
    return <EmptyState>No bracket data available.</EmptyState>;

  const maxChamp = teams[0]?.p_champion || 1;
  const chartData = teams.map((t) => ({
    team: t.team,
    champion: t.p_champion,
  }));

  return (
    <div>
      <h1>Tournament outlook</h1>
      <p className="subtitle">
        Per-team advancement odds from a Monte-Carlo simulation of the remaining
        bracket ({data.n_sims.toLocaleString()} sims, seed {data.seed}). These
        are probabilities across many simulated tournaments — a favourite is
        still far from a lock. Updates whenever the data refetches
        {isFetching ? " (refreshing…)" : ""}.
      </p>

      <div className="grid-2">
        <div className="card">
          <h2>Chance to be champion</h2>
          <ResponsiveContainer width="100%" height={Math.max(220, teams.length * 26)}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ left: 20, right: 40, top: 4, bottom: 4 }}
            >
              <CartesianGrid stroke="#2a3350" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, Math.ceil(maxChamp * 100) / 100]}
                tickFormatter={(v) => pct(v)}
                stroke="#93a0bd"
                fontSize={11}
              />
              <YAxis
                type="category"
                dataKey="team"
                width={110}
                stroke="#93a0bd"
                fontSize={12}
              />
              <Tooltip
                formatter={(v: number) => [pct1(v), "P(champion)"]}
                contentStyle={{
                  background: "#141a2e",
                  border: "1px solid #2a3350",
                  borderRadius: 8,
                  color: "#dfe6f2",
                }}
              />
              <Bar dataKey="champion" radius={[0, 4, 4, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill="#5aa9ff" />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h2>Progression by round</h2>
          <div style={{ overflowX: "auto" }}>
            <table className="fixtures">
              <thead>
                <tr>
                  <th>Team</th>
                  <th className="right">Semi</th>
                  <th className="right">Final</th>
                  <th className="right">Champion</th>
                </tr>
              </thead>
              <tbody>
                {teams.map((t) => (
                  <tr key={t.team_id}>
                    <td className="matchup">{t.team}</td>
                    <td className="right mono">{pct(t.p_reach_semi)}</td>
                    <td className="right mono">{pct(t.p_reach_final)}</td>
                    <td className="right mono">
                      <strong>{pct1(t.p_champion)}</strong>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="dim" style={{ fontSize: 12, marginTop: 10 }}>
            The API returns per-team reach/champion probabilities rather than a
            fixed tie tree, so progression is shown as columns. Reach-round
            probabilities are cumulative (Champion ≤ Final ≤ Semi).
          </p>
        </div>
      </div>
    </div>
  );
}
