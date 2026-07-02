// Sequential colour scale for the score heatmap. Maps a normalised value
// t in [0, 1] to a colour. Uses a perceptual-ish blue -> teal -> yellow ramp
// so higher probability reads as brighter/warmer against the dark theme.

type RGB = [number, number, number];

const STOPS: { t: number; c: RGB }[] = [
  { t: 0.0, c: [23, 28, 42] }, // near background (very low prob)
  { t: 0.2, c: [30, 58, 110] },
  { t: 0.45, c: [24, 116, 155] },
  { t: 0.7, c: [46, 174, 148] },
  { t: 0.88, c: [148, 208, 90] },
  { t: 1.0, c: [245, 221, 66] }, // brightest (highest prob)
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function heatColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const lo = STOPS[i];
    const hi = STOPS[i + 1];
    if (x >= lo.t && x <= hi.t) {
      const local = (x - lo.t) / (hi.t - lo.t || 1);
      const r = Math.round(lerp(lo.c[0], hi.c[0], local));
      const g = Math.round(lerp(lo.c[1], hi.c[1], local));
      const b = Math.round(lerp(lo.c[2], hi.c[2], local));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  const last = STOPS[STOPS.length - 1].c;
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

// Pick readable text colour for a cell given the underlying intensity.
export function heatTextColor(t: number): string {
  return t > 0.6 ? "#0b1020" : "#dfe6f2";
}
