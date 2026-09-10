"""Shared fixtures.

The tests build a small synthetic league rather than reading the real
database, so they are fast, deterministic and do not need the source
archives to be checked out.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import db  # noqa: E402
from plpredict.data.ingest import make_match_id, outcome_of  # noqa: E402


TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """A proper double round-robin, one list of pairings per round.

    Built with the circle method so that every club appears exactly once
    per round - which is the property the point-in-time tests rely on.
    """
    rotation = list(teams)
    half = len(rotation) // 2
    first_half: list[list[tuple[str, str]]] = []
    for round_number in range(len(rotation) - 1):
        pairings = [
            (rotation[position], rotation[-1 - position])
            if round_number % 2 == 0
            else (rotation[-1 - position], rotation[position])
            for position in range(half)
        ]
        first_half.append(pairings)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    reversed_half = [[(away, home) for home, away in rnd] for rnd in first_half]
    return first_half + reversed_half


def _fixture_list(season: str, start: dt.date) -> list[dict]:
    rows = []
    for round_number, pairings in enumerate(_round_robin(TEAMS), start=1):
        date = start + dt.timedelta(days=7 * (round_number - 1))
        for home, away in pairings:
            # Deterministic, with three properties the tests rely on: the
            # club earlier in the alphabet is stronger, playing at home is
            # worth roughly a goal, and draws happen often enough that all
            # three outcomes appear in the training data.
            gap = TEAMS.index(away) - TEAMS.index(home)
            home_goals = min(4, max(0, 1 + gap))
            away_goals = min(4, max(0, 1 - gap))
            if (round_number + TEAMS.index(home)) % 4 == 0:
                away_goals = home_goals  # a regular sprinkling of draws
            rows.append(
                {
                    "match_id": make_match_id(
                        season, "premier_league", home, away, round_number, None
                    ),
                    "season": season,
                    "competition": "premier_league",
                    "matchday": round_number,
                    "stage": None,
                    "match_date": date.isoformat(),
                    "kickoff": "15:00",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "ht_home_goals": None,
                    "ht_away_goals": None,
                    "result": outcome_of(home_goals, away_goals),
                    "status": "played",
                    "source": "test",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )
    return rows


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Five finished synthetic seasons plus one part-played current season."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    rows: list[dict] = []
    seasons = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
    for offset, season in enumerate(seasons):
        rows.extend(_fixture_list(season, dt.date(2021 + offset, 8, 10)))

    # Leave the last two gameweeks of the current season unplayed.
    for row in rows:
        if row["season"] == "2026-27" and row["matchday"] >= 9:
            row.update(
                home_goals=None,
                away_goals=None,
                result=None,
                status="scheduled",
            )

    with db.connect(path) as conn:
        db.upsert_many(conn, "matches", rows, ("match_id",))
    return path
