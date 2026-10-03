"""Rearranged fixtures belong to the gameweek they are played in."""

from __future__ import annotations

import datetime as dt

from plpredict import db
from plpredict.data import gameweeks
from plpredict.data.gameweeks import Fixture, assign
from plpredict.data.ingest import _assign_gameweeks
from plpredict.features.build import FeatureBuilder
from plpredict.web import queries

SEASON = "2030-31"
LEAGUE = "premier_league"


def _day(offset: int) -> dt.date:
    # Saturdays, one week apart, from the opening weekend.
    return dt.date(2030, 8, 10) + dt.timedelta(days=offset)


def _rounds(n_rounds: int, per_round: int = 3) -> list[Fixture]:
    return [
        Fixture(f"r{r}m{m}", r, _day(7 * (r - 1) + (m % 2)))
        for r in range(1, n_rounds + 1)
        for m in range(per_round)
    ]


def test_an_ordinary_season_is_left_as_scheduled():
    fixtures = _rounds(6)
    assert all(assign(fixtures)[f.match_id] == f.matchday for f in fixtures)


def test_a_postponed_match_moves_to_the_gameweek_it_is_played_in():
    fixtures = _rounds(8)
    # Round 2's third match was played in the week of round 6.
    moved = Fixture("r2m2", 2, _day(7 * 5 + 3))
    fixtures = [f for f in fixtures if f.match_id != "r2m2"] + [moved]

    assigned = assign(fixtures)
    assert assigned["r2m2"] == 6
    assert assigned["r2m0"] == 2 and assigned["r2m1"] == 2


def test_a_match_brought_forward_moves_back_a_gameweek():
    fixtures = _rounds(5)
    early = Fixture("r5m2", 5, _day(7 * 2 + 3))  # played in round 3's week
    fixtures = [f for f in fixtures if f.match_id != "r5m2"] + [early]
    assert assign(fixtures)["r5m2"] == 3


def test_the_match_id_survives_a_move(tmp_path):
    """Ids are built from the scheduled round, so stored predictions still join."""
    rows = [
        {"match_id": f.match_id, "season": SEASON, "competition": LEAGUE,
         "matchday": f.matchday, "match_date": f.match_date.isoformat()}
        for f in _rounds(6)
    ]
    rows[3]["match_date"] = _day(7 * 4 + 3).isoformat()  # r2m0 into round 5's week
    _assign_gameweeks(rows)
    moved = rows[3]
    assert moved["match_id"] == "r2m0"
    assert moved["original_matchday"] == 2 and moved["matchday"] == 5
    assert all(r["original_matchday"] == r["matchday"] for r in rows if r is not moved)


# --- against a database -------------------------------------------------


def _insert(conn, match_id, matchday, date, home, away, goals=None, original=None):
    played = goals is not None
    conn.execute(
        """INSERT INTO matches (match_id, season, competition, matchday, original_matchday,
               match_date, kickoff, home_team, away_team, home_goals, away_goals,
               result, status, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, '15:00', ?, ?, ?, ?, ?, ?, '2030-01-01')""",
        (
            match_id, SEASON, LEAGUE, matchday, original or matchday, date.isoformat(),
            home, away,
            goals[0] if played else None, goals[1] if played else None,
            ("H" if goals[0] > goals[1] else "A" if goals[0] < goals[1] else "D") if played else None,
            "played" if played else "scheduled",
        ),
    )


def test_a_postponed_match_does_not_pin_the_next_gameweek(tmp_path):
    with db.connect(tmp_path / "t.db") as conn:
        _insert(conn, "a", 1, _day(0), "A", "B", (1, 0))
        _insert(conn, "b", 1, _day(0), "C", "D")          # postponed, never played
        _insert(conn, "c", 1, _day(1), "E", "F", (2, 2))  # the day after, played
        _insert(conn, "d", 2, _day(7), "A", "C")
        assert gameweeks.next_to_play(conn, SEASON, LEAGUE) == 2


def test_a_round_still_being_played_is_still_next(tmp_path):
    """Saturday's results are in, Sunday's are not: the round is not over."""
    with db.connect(tmp_path / "t.db") as conn:
        _insert(conn, "a", 1, _day(0), "A", "B", (1, 0))
        _insert(conn, "b", 1, _day(1), "C", "D")
        _insert(conn, "c", 2, _day(7), "A", "C")
        assert gameweeks.next_to_play(conn, SEASON, LEAGUE) == 1


def test_the_site_labels_a_rearranged_fixture(tmp_path):
    with db.connect(tmp_path / "t.db") as conn:
        _insert(conn, "a", 5, _day(28), "A", "B")
        _insert(conn, "b", 5, _day(30), "C", "D", original=2)
        page = queries.gameweek(conn, SEASON, 5)
    by_id = {m["match_id"]: m for m in page["matches"]}
    assert by_id["a"]["rearranged_from"] is None
    assert by_id["b"]["rearranged_from"] == 2


def test_a_double_gameweek_second_match_sees_the_first(tmp_path):
    """Rest days count the club's earlier fixture in the same gameweek."""
    with db.connect(tmp_path / "t.db") as conn:
        _insert(conn, "w1a", 1, _day(0), "A", "B", (1, 0))
        _insert(conn, "w1b", 1, _day(0), "C", "D", (0, 0))
        _insert(conn, "w2a", 2, _day(7), "A", "C", (2, 1))   # A plays Saturday...
        _insert(conn, "w2b", 2, _day(7), "B", "D", (1, 1))
        _insert(conn, "w2c", 2, _day(10), "D", "A", (0, 3), original=9)  # ...and Tuesday
        frame = FeatureBuilder(conn).build().set_index("match_id")

    tuesday = frame.loc["w2c"]
    assert tuesday["away_gameweek_fixtures"] == 2.0   # A's double gameweek
    assert tuesday["away_rest_days"] == 3.0           # since Saturday, not since round 1
    assert frame.loc["w2a", "home_gameweek_fixtures"] == 2.0
    assert frame.loc["w2b", "home_gameweek_fixtures"] == 1.0
