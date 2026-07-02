import type { NewsSide } from "../api/types";

/**
 * Small badge for the manual, subjective news-adjustment Elo overlay
 * (injuries/suspensions, spec §5). Visually distinct from the calibrated
 * model output — its own colour + an "adj." tag — so it never reads as a
 * model finding. Renders nothing when there is no delta to show.
 */
export function NewsBadge({
  side,
  label,
}: {
  side: NewsSide | null | undefined;
  /** Optional team name prefix, e.g. for contexts without nearby team labels. */
  label?: string;
}) {
  if (!side || !side.delta) return null;
  const sign = side.delta > 0 ? "+" : "";
  const rounded = Math.round(side.delta);
  const title = side.note
    ? `Manual overlay (subjective, not model output): ${side.note}`
    : "Manual overlay (subjective, not model output)";
  return (
    <span className="tag news" title={title}>
      {label ? `${label}: ` : ""}adj. news {sign}
      {rounded} Elo
    </span>
  );
}
