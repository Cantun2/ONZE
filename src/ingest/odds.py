"""The Odds API client (spec §2.1, §2.2 ``odds`` table).

Fetches live 1N2 (h2h) odds for international soccer and shapes them into rows
for the ``odds(fixture_id, bookmaker, o_home, o_draw, o_away, captured_at)`` table.

The API key is read from the ``ONZE_ODDS_API_KEY`` environment variable and is
**never** hard-coded. It is fine for no key to be set right now: the client and
the ``odds`` table schema must simply exist and fail cleanly (``MissingApiKey``)
when a live call is attempted without a key.

Typical usage::

    from src.ingest.odds import OddsClient
    client = OddsClient()            # raises MissingApiKey only on .fetch_*()
    events = client.fetch_h2h_odds("soccer_fifa_world_cup")
    rows = client.to_odds_rows(events, fixture_id_by_match={...})
"""

from __future__ import annotations

import datetime as _dt
import os

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ENV_KEY = "ONZE_ODDS_API_KEY"

# The Odds API sport key for the World Cup (may change per season; overridable).
DEFAULT_SPORT = "soccer_fifa_world_cup"


class MissingApiKey(RuntimeError):
    """Raised when a live Odds API call is attempted without a configured key."""


class OddsClient:
    """Thin wrapper over The Odds API v4 h2h endpoint.

    Construction never requires network or a key; the key is only demanded when
    a fetching method is actually called, so importing/instantiating is safe in
    environments (CI, tests) without ``ONZE_ODDS_API_KEY`` set.
    """

    def __init__(self, api_key: str | None = None, *, timeout: int = 30) -> None:
        # Do not raise here: allow key-less construction (schema/tests).
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_KEY)
        self.timeout = timeout

    # -- internals ----------------------------------------------------------
    def _require_key(self) -> str:
        if not self.api_key:
            raise MissingApiKey(
                f"No Odds API key: set the {ENV_KEY} environment variable "
                "(see .env.example). Odds fetching is disabled without it."
            )
        return self.api_key

    # -- public -------------------------------------------------------------
    def fetch_h2h_odds(
        self,
        sport: str = DEFAULT_SPORT,
        *,
        regions: str = "eu,uk,us",
        odds_format: str = "decimal",
    ) -> list[dict]:
        """Fetch raw h2h (1N2) odds events for *sport*.

        Returns the API's list of event dicts. Raises :class:`MissingApiKey`
        when no key is configured, and propagates HTTP errors otherwise.
        """
        key = self._require_key()
        url = f"{ODDS_API_BASE}/sports/{sport}/odds"
        params = {
            "apiKey": key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def to_odds_rows(
        events: list[dict],
        fixture_id_by_match: dict[tuple[str, str], int] | None = None,
    ) -> list[dict]:
        """Flatten Odds API events into ``odds`` table rows.

        ``fixture_id_by_match`` maps ``(home_team, away_team)`` -> ``fixture_id``.
        Events without a matching fixture are skipped (odds we can't attach).
        For h2h, the outcome named like the home team -> ``o_home``, the away
        team -> ``o_away``, and "Draw" -> ``o_draw``.
        """
        fixture_id_by_match = fixture_id_by_match or {}
        captured = _dt.datetime.now(_dt.timezone.utc)
        rows: list[dict] = []
        for ev in events:
            home = ev.get("home_team")
            away = ev.get("away_team")
            fixture_id = fixture_id_by_match.get((home, away))
            if fixture_id is None:
                continue
            for book in ev.get("bookmakers", []):
                o_home = o_draw = o_away = None
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name, price = outcome.get("name"), outcome.get("price")
                        if name == home:
                            o_home = price
                        elif name == away:
                            o_away = price
                        elif name and name.lower() == "draw":
                            o_draw = price
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "bookmaker": book.get("key"),
                        "o_home": o_home,
                        "o_draw": o_draw,
                        "o_away": o_away,
                        "captured_at": captured,
                    }
                )
        return rows


if __name__ == "__main__":
    # Smoke: prove the client exists and fails cleanly without a key.
    client = OddsClient()
    if client.api_key:
        data = client.fetch_h2h_odds()
        print(f"fetched {len(data)} events")
    else:
        print(f"No {ENV_KEY} set — client instantiated, live fetch disabled.")
