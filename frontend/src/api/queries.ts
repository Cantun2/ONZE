import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { ServiceUnavailableError } from "./types";

export function useFixtures() {
  return useQuery({
    queryKey: ["fixtures"],
    queryFn: api.fixtures,
  });
}

export function usePrediction(fixtureId: number | null) {
  return useQuery({
    queryKey: ["predict", fixtureId],
    queryFn: () => api.predict(fixtureId as number),
    enabled: fixtureId != null,
  });
}

export function useBracket() {
  return useQuery({
    queryKey: ["bracket"],
    queryFn: api.bracket,
  });
}

export function useBacktest(fromYear?: number) {
  return useQuery({
    queryKey: ["backtest", fromYear ?? null],
    queryFn: () => api.backtest(fromYear),
    // Don't hammer the endpoint while the eval artifact is still pending.
    retry: (failureCount, error) => {
      if (error instanceof ServiceUnavailableError) return false;
      return failureCount < 2;
    },
  });
}
