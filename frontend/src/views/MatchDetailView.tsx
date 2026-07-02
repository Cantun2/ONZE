import { Link, useParams } from "react-router-dom";
import { usePrediction } from "../api/queries";
import type { Prediction } from "../api/types";
import { ErrorState, Loading } from "../components/States";
import { ScoreHeatmap } from "../components/ScoreHeatmap";
import { OneN2Bar } from "../components/OneN2Bar";
import { NewsBadge } from "../components/NewsBadge";
import { pct, pct1 } from "../lib/format";
import { expectedGoals, fmtGoals } from "../lib/matrix";

export function MatchDetailView() {
  const { fixtureId } = useParams();
  const id = fixtureId ? Number(fixtureId) : null;
  const { data, isLoading, error } = usePrediction(id);

  if (id == null || Number.isNaN(id)) return <ErrorState error="Invalid fixture id" />;
  if (isLoading) return <Loading label="Loading prediction…" />;
  if (error) return <ErrorState error={error} />;
  if (!data) return <ErrorState error="No prediction returned." />;

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <Link to="/">← All matches</Link>
      </div>
      <h1>
        {data.home_team} <span className="dim">vs</span> {data.away_team}
      </h1>
      <p className="subtitle">
        {data.stage ?? "—"}
        {data.knockout ? " · knockout tie" : ""} · model {data.model_version}.
        Every number below is derived from the single score-probability matrix —
        the only honest object. No outcome is certain.
      </p>

      <NewsOverlayNote pred={data} />

      <SummaryStrip pred={data} />

      <div className="grid-2">
        <div className="card">
          <h2>Score probability matrix P(x, y)</h2>
          <ScoreHeatmap
            matrix={data.matrix}
            homeTeam={data.home_team}
            awayTeam={data.away_team}
          />
        </div>

        <div>
          <MarketsPanel pred={data} />
        </div>
      </div>
    </div>
  );
}

/**
 * Surfaces the live per-team news-adjustment Elo overlay (injuries/
 * suspensions, spec §5): a manual, subjective nudge applied at
 * prediction-time only — never part of the calibrated model. Renders
 * nothing when neither side has a nonzero delta, so it never implies an
 * adjustment where none was made.
 */
function NewsOverlayNote({ pred }: { pred: Prediction }) {
  const na = pred.news_adjustment;
  if (!na) return null;
  const hasNews = na.home.delta !== 0 || na.away.delta !== 0;
  if (!hasNews) return null;
  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <div className="market-label" style={{ marginBottom: 8 }}>
        Manual news overlay
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <NewsBadge side={na.home} label={pred.home_team} />
        <NewsBadge side={na.away} label={pred.away_team} />
      </div>
      {na.home.delta !== 0 && na.home.note && (
        <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>
          {pred.home_team}: {na.home.note}
        </div>
      )}
      {na.away.delta !== 0 && na.away.note && (
        <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>
          {pred.away_team}: {na.away.note}
        </div>
      )}
      <div className="caveat" role="note">
        News adjustment is a manual, subjective Elo nudge for injuries/
        suspensions — not part of the calibrated model.
      </div>
    </div>
  );
}

/**
 * A compact headline strip: the projected (expected-goals) scoreline and the
 * 1N2 distribution side by side. This foregrounds the signal — two evenly
 * matched ties that both have a 1-1 mode still read differently here because
 * their xG and win bars differ.
 */
function SummaryStrip({ pred }: { pred: Prediction }) {
  const { xgHome, xgAway } = expectedGoals(pred.matrix);
  return (
    <div
      className="card"
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 28,
        alignItems: "center",
        marginBottom: 18,
      }}
    >
      <div>
        <div className="dim" style={{ fontSize: 12 }}>
          Projected scoreline · expected goals (model)
        </div>
        <div style={{ fontSize: 22, fontWeight: 700 }} className="mono">
          {pred.home_team} {fmtGoals(xgHome)}{" "}
          <span className="dim">–</span> {fmtGoals(xgAway)} {pred.away_team}
        </div>
        <div className="dim" style={{ fontSize: 11 }}>
          mean goals per side — not a prediction of the actual score. Modal score{" "}
          {pred.most_likely_score}.
        </div>
      </div>
      <div style={{ minWidth: 240 }}>
        <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
          Match result (90&apos;)
        </div>
        <OneN2Bar
          pHome={pred.p_home}
          pDraw={pred.p_draw}
          pAway={pred.p_away}
          showLegend
        />
      </div>
    </div>
  );
}

function MarketsPanel({ pred }: { pred: Prediction }) {
  const ouLines = Object.keys(pred.over_under).sort(
    (a, b) => Number(a) - Number(b),
  );

  return (
    <div className="card">
      <h2>Derived markets</h2>

      <div className="market-block">
        <div className="market-label" style={{ marginBottom: 6 }}>
          Match result (90&apos;)
        </div>
        <OneN2Bar
          pHome={pred.p_home}
          pDraw={pred.p_draw}
          pAway={pred.p_away}
          showLegend
          width={undefined}
        />
      </div>

      {pred.knockout && pred.advance && (
        <div className="market-block">
          <div className="market-label" style={{ marginBottom: 6 }}>
            Advances to next round (incl. extra time + penalties)
          </div>
          <div className="market-row">
            <span>{pred.home_team}</span>
            <span className="mono">{pct1(pred.advance.p_home_advance)}</span>
          </div>
          <div className="market-row">
            <span>{pred.away_team}</span>
            <span className="mono">{pct1(pred.advance.p_away_advance)}</span>
          </div>
        </div>
      )}

      <div className="market-block">
        <div className="market-label" style={{ marginBottom: 6 }}>
          Over / Under (total goals)
        </div>
        {ouLines.map((line) => {
          const ou = pred.over_under[line];
          return (
            <div className="market-row" key={line}>
              <span className="market-label">Line {line}</span>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="probbar">
                  <span
                    style={{ width: `${ou.over * 100}%`, background: "var(--home)" }}
                  />
                  <span
                    style={{ width: `${ou.under * 100}%`, background: "var(--away)" }}
                  />
                </span>
                <span className="mono">
                  O {pct(ou.over)} / U {pct(ou.under)}
                </span>
              </span>
            </div>
          );
        })}
      </div>

      <div className="market-block">
        <div className="market-row">
          <span className="market-label">Both teams to score</span>
          <span className="mono">
            Yes {pct1(pred.btts)} · No {pct1(1 - pred.btts)}
          </span>
        </div>
      </div>

      <div className="market-block">
        <div className="market-label" style={{ marginBottom: 6 }}>
          Top 5 scorelines
        </div>
        <ul className="top5">
          {pred.top5_scores.map((s, i) => (
            <li key={i}>
              <span className="mono">
                {s.home}–{s.away}
              </span>
              <span className="mono">{pct1(s.prob)}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="market-block">
        <div className="market-label" style={{ marginBottom: 6 }}>
          Bookmaker overlay
        </div>
        {/* Odds are not yet served by the API. Slot reserved so the book's
            implied (de-vigged) probability can be overlaid later without a
            redesign. It is intentionally empty, not mocked. */}
        <div className="slot-empty">
          No live odds available yet. When the API serves odds, the book&apos;s
          implied probability will appear here for comparison.
        </div>
      </div>
    </div>
  );
}
