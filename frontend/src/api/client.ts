import type {
  Backtest,
  Bracket,
  Fixture,
  Prediction,
} from "./types";
import { ServiceUnavailableError } from "./types";

// API base URL is configurable via the Vite env var VITE_API_BASE_URL.
// Defaults to the local FastAPI dev server.
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: "application/json" },
    });
  } catch (err) {
    throw new Error(
      `Cannot reach the ONZE API at ${API_BASE_URL}. Is the backend running? (${
        (err as Error).message
      })`,
    );
  }

  if (res.status === 503) {
    // Surface a typed error so callers can show a "pending" state.
    let detail = "Service not available yet.";
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore body parse errors */
    }
    throw new ServiceUnavailableError(detail);
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(`API error ${res.status}: ${detail}`);
  }

  return (await res.json()) as T;
}

export const api = {
  fixtures: () => getJSON<Fixture[]>("/fixtures"),
  predict: (fixtureId: number) => getJSON<Prediction>(`/predict/${fixtureId}`),
  bracket: () => getJSON<Bracket>("/bracket"),
  backtest: (fromYear?: number) =>
    getJSON<Backtest>(
      `/eval/backtest${fromYear != null ? `?from=${fromYear}` : ""}`,
    ),
};
