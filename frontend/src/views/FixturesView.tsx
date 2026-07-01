import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useFixtures } from "../api/queries";
import type { Fixture } from "../api/types";
import { ErrorState, Loading, EmptyState } from "../components/States";
import { OneN2Bar } from "../components/OneN2Bar";
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
      <h1>Remaining matches</h1>
      <p className="subtitle">
        One row per match. The 1N2 bar and qualification figures are full
        probability distributions — they express uncertainty, not a called
        result. Click a row for the full score matrix.
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
              <th>Most likely</th>
              <th>Outcome (1N2)</th>
              <th>Qualification</th>
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

function FixtureRow({ f, onOpen }: { f: Fixture; onOpen: () => void }) {
  const { data: pred } = usePrediction(f.fixture_id);

  const conf = pred
    ? confidenceLabel(pred.p_home, pred.p_draw, pred.p_away)
    : null;

  return (
    <tr
      className="row"
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
        {f.knockout && <span className="tag ko">KO</span>}
      </td>
      <td className="matchup">
        {f.home_team}
        <span className="vs">vs</span>
        {f.away_team}
      </td>
      <td className="mono">{pred ? pred.most_likely_score : "—"}</td>
      <td>
        {pred ? (
          <OneN2Bar pHome={pred.p_home} pDraw={pred.p_draw} pAway={pred.p_away} />
        ) : (
          <span className="dim">—</span>
        )}
        {conf && (
          <div className={`conf ${conf.level}`} style={{ marginTop: 4 }}>
            <span className="dot" />
            {conf.label}
          </div>
        )}
      </td>
      <td className="mono">
        {f.knockout ? (
          pred?.advance ? (
            <span title="Chance to advance (incl. extra time + penalties)">
              {f.home_team.slice(0, 3).toUpperCase()} {pct(pred.advance.p_home_advance)}
              {" · "}
              {f.away_team.slice(0, 3).toUpperCase()} {pct(pred.advance.p_away_advance)}
            </span>
          ) : (
            <span className="dim">—</span>
          )
        ) : (
          <span className="dim">group / n/a</span>
        )}
      </td>
      <td>
        <span className="dim">open →</span>
      </td>
    </tr>
  );
}
