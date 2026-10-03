"""Which gameweek a fixture is actually played in.

The fixture list numbers every match by the round it was *scheduled*
in, and keeps that number when a match is moved. A match postponed from
matchday 8 and played the following April is still "matchday 8" in the
source, which is wrong for everything this project uses a gameweek for:

- Features for a gameweek are computed from the state before its first
  kick-off, and its results are fed in afterwards. A matchday-8 fixture
  played in April would be featurised from September's state, and its
  April result would be fed into every team's form, table and Elo from
  September onward — a result from the future, visible to seven months
  of training rows.
- "The next gameweek" is the lowest round with a match still to play,
  so one postponed fixture would pin the site to that round until it was
  rearranged, and the rounds after it would never be forecast in time.

Fantasy players know the fix as double and blank gameweeks: a moved
fixture belongs to the round in which it is played. That is what this
assigns. Each round's *core* is the cluster of dates on which most of
its matches are played; the rounds' cores, in date order, cut the season
into consecutive windows, and every match belongs to the window its
date falls in. A club with a rearranged fixture in a window plays twice
in that gameweek, and a club whose match moved out of one plays none.

This needs nothing but dates, so it applies identically to every season
in the archive and to the live one.
"""

from __future__ import annotations

import bisect
import datetime as dt
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

# How far from the round's median date a match can be and still count as
# played with its round. A normal round runs Friday to Monday; anything
# further out was moved.
CORE_DAYS = 4


@dataclass(frozen=True)
class Fixture:
    match_id: str
    matchday: int
    match_date: dt.date


def assign(fixtures: Iterable[Fixture]) -> dict[str, int]:
    """Map each fixture of one season to the gameweek it is played in."""
    by_round: dict[int, list[Fixture]] = defaultdict(list)
    for fixture in fixtures:
        by_round[fixture.matchday].append(fixture)
    if not by_round:
        return {}

    core_start: dict[int, dt.date] = {}
    in_core: set[str] = set()
    for matchday, group in by_round.items():
        dates = sorted(fixture.match_date for fixture in group)
        median = dates[(len(dates) - 1) // 2]
        core = [f for f in group if abs((f.match_date - median).days) <= CORE_DAYS]
        core_start[matchday] = min(f.match_date for f in core)
        in_core.update(f.match_id for f in core)

    # Windows run from one round's core to the next, in date order.
    ordered = sorted(core_start, key=lambda matchday: (core_start[matchday], matchday))
    starts = [core_start[matchday] for matchday in ordered]

    assigned: dict[str, int] = {}
    for matchday, group in by_round.items():
        for fixture in group:
            if fixture.match_id in in_core:
                assigned[fixture.match_id] = matchday
                continue
            position = bisect.bisect_right(starts, fixture.match_date) - 1
            assigned[fixture.match_id] = ordered[max(position, 0)]
    return assigned


# An unplayed match counts as postponed once any match dated after it has
# a result. A round whose results are still arriving is not affected:
# there the unplayed matches are the later ones, not the earlier.
_NEXT_SQL = """
    SELECT MIN(m.matchday) FROM matches m
    WHERE m.season = ? AND m.competition = ? AND m.status != 'played'
      AND NOT EXISTS (
        SELECT 1 FROM matches later
        WHERE later.season = m.season AND later.competition = m.competition
          AND later.status = 'played'
          AND later.match_date > m.match_date
      )
"""


def next_to_play(conn: sqlite3.Connection, season: str, competition: str) -> int | None:
    """The gameweek about to be played: the lowest with a fixture still to come.

    A match left unplayed while later fixtures have results was postponed.
    It stays unplayed in the source until it is rearranged, and until then
    it must not count - otherwise one postponement pins "next" to its
    gameweek for weeks and nothing after it is forecast in time. Once it
    is given a new date, ``assign`` moves it to the gameweek of that date
    and it is predicted with that gameweek.

    Shared by the pipeline, the site and the publication check, so all
    three always agree on which gameweek is current.
    """
    row = conn.execute(_NEXT_SQL, (season, competition)).fetchone()
    return int(row[0]) if row and row[0] is not None else None
