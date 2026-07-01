import { pct } from "../lib/format";

/** Small stacked 1N2 bar: home win / draw / away win. */
export function OneN2Bar({
  pHome,
  pDraw,
  pAway,
  showLegend = false,
  width,
}: {
  pHome: number;
  pDraw: number;
  pAway: number;
  showLegend?: boolean;
  width?: number;
}) {
  const total = pHome + pDraw + pAway || 1;
  const title = `Home ${pct(pHome)} · Draw ${pct(pDraw)} · Away ${pct(pAway)}`;
  return (
    <div>
      <div
        className="bar1n2"
        style={width ? { width } : undefined}
        title={title}
        role="img"
        aria-label={title}
      >
        <span className="seg-home" style={{ width: `${(pHome / total) * 100}%` }} />
        <span className="seg-draw" style={{ width: `${(pDraw / total) * 100}%` }} />
        <span className="seg-away" style={{ width: `${(pAway / total) * 100}%` }} />
      </div>
      {showLegend && (
        <div className="bar1n2-legend">
          <span>
            <span className="swatch" style={{ background: "var(--home)" }} />
            Home {pct(pHome)}
          </span>
          <span>
            <span className="swatch" style={{ background: "var(--draw)" }} />
            Draw {pct(pDraw)}
          </span>
          <span>
            <span className="swatch" style={{ background: "var(--away)" }} />
            Away {pct(pAway)}
          </span>
        </div>
      )}
    </div>
  );
}
