"""Player availability from the Fantasy Premier League API.

The API reports every Premier League player's availability as it stands
right now: a status (available, doubtful, injured, suspended,
unavailable), a percentage chance of playing the next round, and the
news line behind it ("Hamstring injury - 75% chance of playing"). It is
the most complete free injury feed there is, and it has one serious
limitation: it has no history. Once a player recovers, the record that
he was ever doubtful is gone, and nobody archives it.

A model can only learn from what was known before kick-off, so the one
thing that cannot be reconstructed later is exactly the thing needed.
The only remedy is to start writing it down. Every run fetches the
current picture and appends to a log, and only what changed since the
last observation is written: a status moving from doubtful to injured,
a percentage changing, a new news line, a transfer. Availability as it
stood at any moment is then the last logged row per player at or before
that moment, which is what the feature layer will read.

The log is committed to the repository, because the working database is
rebuilt from source on every run and a CI runner starts without one.
Each season starts with a full observation of every player, so a season
can be read on its own without reaching back into the previous one.
"""

from __future__ import annotations

import csv
import datetime as dt
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

_POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# A change in any of these is a change in availability worth recording.
# ``news_added`` is kept as data but is not one of them: it moves when a
# club merely re-confirms an existing line, which says nothing new.
_TRACKED = ("fpl_team", "status", "chance_next", "chance_this", "news")


@dataclass(frozen=True)
class Availability:
    """One player's availability, as the API reported it at one moment."""

    observed_at: str
    season: str
    fpl_next_event: str
    player_code: str
    player_id: str
    web_name: str
    position: str
    fpl_team: str
    status: str
    chance_next: str
    chance_this: str
    news: str
    news_added: str


LOG_COLUMNS = tuple(Availability.__dataclass_fields__)


def fetch_bootstrap(
    url: str = BOOTSTRAP_URL, attempts: int = 3, timeout: float = 30.0
) -> dict[str, Any]:
    """Fetch the API's main payload, retrying a transient failure.

    The API answers with an HTML holding page rather than JSON while the
    game is being updated, which happens around each gameweek deadline.
    That is treated like any other failed attempt.
    """
    headers = {"User-Agent": "plpredict (football research project)"}
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "elements" not in payload:
                raise ValueError("response has no player list")
            return payload
        except (requests.RequestException, ValueError) as error:
            last = error
            if attempt < attempts:
                time.sleep(10 * attempt)
    raise RuntimeError(f"FPL API unavailable after {attempts} attempts: {last}")


def _text(value: object) -> str:
    """Log values are strings, so an absent value and an empty one agree."""
    return "" if value is None else str(value)


def parse_bootstrap(
    payload: dict[str, Any], season: str, observed_at: dt.datetime
) -> list[Availability]:
    teams = {team["id"]: team["name"] for team in payload["teams"]}
    upcoming = next(
        (event["id"] for event in payload.get("events", []) if event.get("is_next")),
        None,
    )
    stamp = observed_at.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    players = []
    for element in payload["elements"]:
        position = _POSITIONS.get(element["element_type"])
        if position is None:
            # Not a player: the game has at times listed managers too.
            continue
        players.append(
            Availability(
                observed_at=stamp,
                season=season,
                fpl_next_event=_text(upcoming),
                player_code=_text(element["code"]),
                player_id=_text(element["id"]),
                web_name=_text(element["web_name"]),
                position=position,
                fpl_team=teams[element["team"]],
                status=_text(element["status"]),
                chance_next=_text(element.get("chance_of_playing_next_round")),
                chance_this=_text(element.get("chance_of_playing_this_round")),
                news=_text(element.get("news")).strip(),
                news_added=_text(element.get("news_added")),
            )
        )
    return players


def read_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_status(
    rows: Iterable[dict[str, str]], season: str, as_of: str | None = None
) -> dict[str, dict[str, str]]:
    """Each player's availability as it stood at ``as_of`` (UTC ISO), by player code.

    With no ``as_of`` this is the most recent observation of each player.
    Timestamps are fixed-width UTC strings, so they compare as text.
    """
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["season"] != season:
            continue
        if as_of is not None and row["observed_at"] > as_of:
            continue
        current = latest.get(row["player_code"])
        if current is None or row["observed_at"] >= current["observed_at"]:
            latest[row["player_code"]] = row
    return latest


def record(snapshot: list[Availability], log_path: Path) -> int:
    """Append whatever in ``snapshot`` differs from the log. Returns rows written."""
    if not snapshot:
        return 0
    season = snapshot[0].season
    known = latest_status(read_log(log_path), season)

    changed = []
    for player in snapshot:
        row = asdict(player)
        previous = known.get(player.player_code)
        if previous is None or any(previous[key] != row[key] for key in _TRACKED):
            changed.append(row)

    if not changed:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.is_file() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(changed)
    return len(changed)
