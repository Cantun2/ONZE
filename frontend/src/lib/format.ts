// Small formatting helpers. Probabilities are always shown as honest
// percentages with modest precision — no "sure thing" rounding to 100%.

export function pct(p: number, digits = 0): string {
  if (p == null || Number.isNaN(p)) return "–";
  return `${(p * 100).toFixed(digits)}%`;
}

export function pct1(p: number): string {
  return pct(p, 1);
}

/**
 * A qualitative confidence label derived from how peaked the outcome
 * distribution is (the max of the three 1N2 probabilities). This is a *display*
 * cue, not a claim of certainty: even the top label stays hedged.
 */
export function confidenceLabel(pHome: number, pDraw: number, pAway: number): {
  label: string;
  level: "low" | "medium" | "high";
} {
  const top = Math.max(pHome, pDraw, pAway);
  if (top >= 0.6) return { label: "Leans clear", level: "high" };
  if (top >= 0.45) return { label: "Slight lean", level: "medium" };
  return { label: "Wide open", level: "low" };
}

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
