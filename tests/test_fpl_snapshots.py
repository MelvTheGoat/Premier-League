"""The FPL availability log: only changes are written, and the past can be read back."""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from plpredict.data.sources import fpl

SEASON = "2026-27"


def _payload() -> dict:
    return {
        "teams": [{"id": 1, "name": "Man City"}, {"id": 2, "name": "Spurs"}],
        "events": [
            {"id": 5, "is_current": True, "is_next": False},
            {"id": 6, "is_current": False, "is_next": True},
        ],
        "elements": [
            {
                "code": 1001, "id": 11, "web_name": "Keeper", "element_type": 1,
                "team": 1, "status": "a", "chance_of_playing_next_round": None,
                "chance_of_playing_this_round": None, "news": "", "news_added": None,
            },
            {
                "code": 1002, "id": 12, "web_name": "Striker", "element_type": 4,
                "team": 2, "status": "d", "chance_of_playing_next_round": 75,
                "chance_of_playing_this_round": 75,
                "news": 'Hamstring injury, "assessed daily" - 75% chance of playing',
                "news_added": "2026-09-24T10:00:00Z",
            },
            {
                # Not a player; the game has listed managers in some seasons.
                "code": 9999, "id": 99, "web_name": "Boss", "element_type": 5,
                "team": 1, "status": "a", "news": "",
            },
        ],
    }


def _at(day: int, hour: int = 6) -> dt.datetime:
    return dt.datetime(2026, 9, day, hour, tzinfo=dt.timezone.utc)


@pytest.fixture
def log(tmp_path):
    return tmp_path / "snapshots" / "fpl_availability.csv"


def test_parsing_keeps_players_and_drops_everything_else():
    players = fpl.parse_bootstrap(_payload(), SEASON, _at(30))
    assert [p.web_name for p in players] == ["Keeper", "Striker"]

    keeper, striker = players
    assert keeper.position == "GKP" and striker.position == "FWD"
    assert striker.fpl_team == "Spurs"
    assert striker.status == "d" and striker.chance_next == "75"
    # No figure from the API is an empty field, not the string "None".
    assert keeper.chance_next == "" and keeper.news_added == ""
    assert keeper.fpl_next_event == "6"
    assert keeper.observed_at == "2026-09-30T06:00:00Z"


def test_the_first_run_records_every_player(log):
    written = fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)
    assert written == 2
    assert log.read_text(encoding="utf-8").splitlines()[0] == ",".join(fpl.LOG_COLUMNS)


def test_an_unchanged_day_writes_nothing(log):
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)
    before = log.read_bytes()

    assert fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(25)), log) == 0
    assert log.read_bytes() == before


def test_only_the_player_whose_availability_changed_is_written(log):
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)

    payload = _payload()
    striker = payload["elements"][1]
    striker.update(status="i", chance_of_playing_next_round=0, news="Hamstring injury - Expected back 25 Oct")
    assert fpl.record(fpl.parse_bootstrap(payload, SEASON, _at(26)), log) == 1

    rows = fpl.read_log(log)
    assert len(rows) == 3
    assert rows[-1]["web_name"] == "Striker" and rows[-1]["status"] == "i"


def test_a_reconfirmed_news_line_is_not_a_change(log):
    """Clubs re-date the same line; availability has not moved."""
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)

    payload = _payload()
    payload["elements"][1]["news_added"] = "2026-09-26T09:00:00Z"
    assert fpl.record(fpl.parse_bootstrap(payload, SEASON, _at(26)), log) == 0


def test_a_transfer_between_clubs_is_a_change(log):
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)

    payload = _payload()
    payload["elements"][0]["team"] = 2
    assert fpl.record(fpl.parse_bootstrap(payload, SEASON, _at(26)), log) == 1


def test_availability_can_be_read_as_it_stood_at_any_moment(log):
    """The whole point of the log: what was known before a given kick-off."""
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)
    payload = _payload()
    payload["elements"][1].update(status="a", chance_of_playing_next_round=None, news="")
    fpl.record(fpl.parse_bootstrap(payload, SEASON, _at(27)), log)

    rows = fpl.read_log(log)
    before = fpl.latest_status(rows, SEASON, as_of="2026-09-26T12:00:00Z")
    after = fpl.latest_status(rows, SEASON, as_of="2026-09-28T12:00:00Z")
    too_early = fpl.latest_status(rows, SEASON, as_of="2026-09-01T00:00:00Z")

    assert before["1002"]["status"] == "d"
    assert after["1002"]["status"] == "a"
    assert too_early == {}


def test_a_new_season_starts_with_a_full_observation(log):
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)
    assert fpl.record(fpl.parse_bootstrap(_payload(), "2027-28", _at(28)), log) == 2


def test_news_with_commas_and_quotes_survives_the_round_trip(log):
    fpl.record(fpl.parse_bootstrap(_payload(), SEASON, _at(24)), log)
    rows = fpl.read_log(log)
    assert rows[1]["news"] == 'Hamstring injury, "assessed daily" - 75% chance of playing'


def test_a_holding_page_is_retried_then_reported(monkeypatch):
    """Around each deadline the API serves HTML while the game updates."""

    class HoldingPage:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("Expecting value")

    calls = []
    monkeypatch.setattr(fpl.requests, "get", lambda *a, **k: calls.append(1) or HoldingPage())
    monkeypatch.setattr(fpl.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fpl.fetch_bootstrap("https://example.test", attempts=3)
    assert len(calls) == 3


def test_the_log_is_unchanged_by_a_payload_it_has_already_seen(log):
    payload = _payload()
    fpl.record(fpl.parse_bootstrap(copy.deepcopy(payload), SEASON, _at(24)), log)
    fpl.record(fpl.parse_bootstrap(copy.deepcopy(payload), SEASON, _at(24, 18)), log)
    assert len(fpl.read_log(log)) == 2
