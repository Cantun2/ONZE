import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useFixtures } from "../api/queries";
import type { Fixture } from "../api/types";
import { ErrorState, Loading, EmptyState } from "../components/States";
import { OneN2Bar } from "../components/OneN2Bar";
import { NewsBadge } from "../components/NewsBadge";
import { formatDate } from "../lib/format";

type SortKey = "date" | "stage";

export function FixturesView() {
  const { data, isLoading, error } = useFixtures();
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [asc, setAsc] = useState(true);
  const navigate = useNavigate();

  const sorted = useMemo(() => {
    if (!data) return [];
    const copy = [...data];
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "date") {
        cmp = (a.date ?? "").localeCompare(b.date ?? "");
      } else {
        cmp = (a.stage ?? "").localeCompare(b.stage ?? "");
      }
      if (cmp === 0) cmp = a.fixture_id - b.fixture_id;
      return asc ? cmp : -cmp;
    });
    return copy;
  }, [data, sortKey, asc]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) setAsc((v) => !v);
    else {
      setSortKey(key);
      setAsc(true);
    }
  }

  if (isLoading) return <Loading label="Loading fixtures…" />;
  if (error) return <ErrorState error={error} />;
  if (!data || data.length === 0)
    return <EmptyState>No remaining fixtures. Try POST /refresh on the API.</EmptyState>;

  const arrow = (key: SortKey) =>
    sortKey === key ? (asc ? " ▲" : " ▼") : "";

  return (
    <div>
      <h1>Matches</h1>
      <p className="subtitle">
        One row per tie. Matches already played show the <strong>actual
        result</strong> — a fact, marked <span className="tag final">final</span> —
        with the model's pre-match view kept as a muted secondary line. Matches
        still to play lead with the model favourite and its win/advance
        probability: the 1N2 bar and projected (expected) goals are full
        distributions — they express uncertainty, not a called result. Click a
        row for the full score matrix.
      </p>

      <div className="card" style={{ padding: 0, overflowX: "auto" }}>
        <table className="fixtures">
          <thead>
            <tr>
              <th
                className="sortable"
                onClick={() => toggleSort("date")}
                aria-sort={sortKey === "date" ? (asc ? "ascending" : "descending") : "none"}
              >
                Date{arrow("date")}
              </th>
              <th
                className="sortable"
                onClick={() => toggleSort("stage")}
                aria-sort={sortKey === "stage" ? (asc ? "ascending" : "descending") : "none"}
              >
                Stage{arrow("stage")}
              </th>
              <th>Match</th>
              <th>Model favourite</th>
              <th>Outcome (1N2)</th>
              <th>Projection</th>
              <th>Read</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((f) => (
              <FixtureRow key={f.fixture_id} f={f} onOpen={() => navigate(`/match/${f.fixture_id}`)} />
            ))}
          </tbody>
        </table>
      </div>

      <div className="bar1n2-legend" style={{ marginTop: 12 }}>
        <span>
          <span className="swatch" style={{ background: "var(--home)" }} />
          Home win
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--draw)" }} />
          Draw (90&apos;)
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--away)" }} />
          Away win
        </span>
      </div>
    </div>
  );
}

/**
 * Each row lazily fetches its own prediction so the table shows the most-likely
 * score, 1N2 bar, a confidence cue and (for knockout ties) qualification %.
 * The prediction fetch is the real API path; while pending we show a dash.
 */
import { usePrediction } from "../api/queries";
import { confidenceLabel, pct } from "../lib/format";
import { expectedGoals, fmtGoals } from "../lib/matrix";

/**
 * Favourite = the side (or draw) with the highest outcome probability. For KO
 * ties we prefer the advance probabilities, since "who goes through" is the
 * headline; for the modal-draw case in regulation we fall back to 1N2.
 */
function favourite(
  f: Fixture,
  pred: NonNullable<ReturnType<typeof usePrediction>["data"]>,
): { name: string; prob: number; kind: "advance" | "win" | "draw" } {
  if (f.knockout && pred.advance) {
    return pred.advance.p_home_advance >= pred.advance.p_away_advance
      ? { name: f.home_team, prob: pred.advance.p_home_advance, kind: "advance" }
      : { name: f.away_team, prob: pred.advance.p_away_advance, kind: "advance" };
  }
  const outcomes = [
    { name: f.home_team, prob: pred.p_home, kind: "win" as const },
    { name: "Draw (90')", prob: pred.p_draw, kind: "draw" as const },
    { name: f.away_team, prob: pred.p_away, kind: "win" as const },
  ];
  return outcomes.reduce((a, b) => (b.prob > a.prob ? b : a));
}

function FixtureRow({ f, onOpen }: { f: Fixture; onOpen: () => void }) {
  const { data: pred } = usePrediction(f.fixture_id);

  const conf = pred
    ? confidenceLabel(pred.p_home, pred.p_draw, pred.p_away)
    : null;
  const fav = pred ? favourite(f, pred) : null;
  const xg = pred ? expectedGoals(pred.matrix) : null;
  const decided = Boolean(f.played && f.result);
  const hasNews =
    pred?.news_adjustment &&
    (pred.news_adjustment.home.delta !== 0 || pred.news_adjustment.away.delta !== 0);

  return (
    <tr
      className={`row${decided ? " decided" : ""}`}
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
    >
      <td className="mono">{formatDate(f.date)}</td>
      <td>
        {f.stage ?? "—"}{" "}
        {f.knockout && <span className="tag ko">KO</span>}{" "}
        {decided && <span className="tag final">Final</span>}
      </td>
      <td className="matchup">
        {decided && f.result ? (
          <DecidedMatchup f={f} result={f.result} />
        ) : (
          <>
            {f.home_team}
            <span className="vs">vs</span>
            {f.away_team}
          </>
        )}
        {hasNews && pred?.news_adjustment && (
          <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap" }}>
            <NewsBadge side={pred.news_adjustment.home} />
            <NewsBadge side={pred.news_adjustment.away} />
          </div>
        )}
      </td>
      <td className={decided ? "pre-match" : undefined}>
        {decided && (
          <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>
            model pre-match:
          </div>
        )}
        {fav ? (
          <div>
            <div style={{ fontWeight: 700, fontSize: 17 }} className="mono">
              {pct(fav.prob)}
            </div>
            <div className="dim" style={{ fontSize: 12 }}>
              {fav.kind === "advance"
                ? `${fav.name} to advance`
                : fav.kind === "draw"
                  ? "level (draw most likely)"
                  : `${fav.name} to win`}
            </div>
          </div>
        ) : (
          <span className="dim">—</span>
        )}
      </td>
      <td className={decided ? "pre-match" : undefined}>
        {pred ? (
          <OneN2Bar pHome={pred.p_home} pDraw={pred.p_draw} pAway={pred.p_away} />
        ) : (
          <span className="dim">—</span>
        )}
        {conf && !decided && (
          <div className={`conf ${conf.level}`} style={{ marginTop: 4 }}>
            <span className="dot" />
            {conf.label}
          </div>
        )}
      </td>
      <td className={decided ? "pre-match" : undefined}>
        {decided && (
          <div className="dim" style={{ fontSize: 11, marginBottom: 2 }}>
            model pre-match xG:
          </div>
        )}
        {pred && xg ? (
          <div>
            <div
              className="mono"
              title="Expected goals (model): mean goals for each side, not a predicted actual score."
            >
              {fmtGoals(xg.xgHome)}{" "}
              <span className="dim">–</span> {fmtGoals(xg.xgAway)}{" "}
              <span className="dim" style={{ fontSize: 11 }}>
                proj. xG
              </span>
            </div>
            <span
              className="tag"
              style={{ marginTop: 4 }}
              title="Single most likely exact scoreline (mode of the distribution)"
            >
              modal {pred.most_likely_score}
            </span>
          </div>
        ) : (
          <span className="dim">—</span>
        )}
      </td>
      <td>
        <span className="dim">{decided ? "view →" : "open →"}</span>
      </td>
    </tr>
  );
}

/** Actual result for a decided tie: winner bold, loser muted, score front and
 * centre, with a small shootout badge when the 90'/ET score was level. */
function DecidedMatchup({
  f,
  result,
}: {
  f: Fixture;
  result: NonNullable<Fixture["result"]>;
}) {
  const homeWon = result.winner_id === f.home_id;
  const awayWon = result.winner_id === f.away_id;
  return (
    <div className="result-line">
      <span className={homeWon ? "winner" : "loser"}>{f.home_team}</span>{" "}
      <span className="mono">
        {result.home_goals}–{result.away_goals}
      </span>{" "}
      <span className={awayWon ? "winner" : "loser"}>{f.away_team}</span>{" "}
      {result.shootout && (
        <span className="tag pens" title="Decided on penalties after extra time">
          pens
        </span>
      )}
    </div>
  );
}
