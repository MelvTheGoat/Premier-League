"""The point-in-time guarantee is the property worth testing hardest.

If a feature can see a result from its own gameweek, every accuracy
number the platform reports is inflated and nothing else matters.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from plpredict import db
from plpredict.features.build import FeatureBuilder, feature_columns
from plpredict.features.elo import EloTable
from plpredict.features.state import MatchOutcome, TeamState
from plpredict.config import FeatureConfig


def _build(seeded_db):
    with db.connect(seeded_db) as conn:
        return FeatureBuilder(conn).build()


def test_first_ever_match_has_no_form(seeded_db):
    frame = _build(seeded_db)
    opener = frame.sort_values(["season", "matchday"]).iloc[0]
    assert opener["home_form6_games"] == 0
    assert pd.isna(opener["home_form6_ppg"])
    assert opener["home_table_points"] == 0


def test_table_reflects_only_earlier_gameweeks(seeded_db):
    frame = _build(seeded_db)
    season = frame[frame["season"] == "2025-26"]
    for matchday in sorted(season["matchday"].unique()):
        rows = season[season["matchday"] == matchday]
        # Points on the table can only come from gameweeks already played.
        assert (rows["home_table_played"] <= matchday - 1).all()
        assert (rows["away_table_played"] <= matchday - 1).all()


def test_every_match_in_a_gameweek_sees_the_same_table(seeded_db):
    """A Monday fixture must not be featurised using Saturday's results."""
    frame = _build(seeded_db)
    season = frame[frame["season"] == "2025-26"]
    for matchday in sorted(season["matchday"].unique()):
        rows = season[season["matchday"] == matchday]
        played = set(rows["home_table_played"]) | set(rows["away_table_played"])
        assert len(played) == 1, f"GW{matchday} rows disagree on the table"


def test_form_window_never_exceeds_its_size(seeded_db):
    frame = _build(seeded_db)
    for window in (4, 6, 10):
        assert (frame[f"home_form{window}_games"] <= window).all()


def test_unplayed_fixtures_are_featurised_without_a_result(seeded_db):
    frame = _build(seeded_db)
    upcoming = frame[frame["status"] == "scheduled"]
    assert not upcoming.empty
    assert upcoming["result"].isna().all()
    assert upcoming["elo_expected_home"].notna().all()


def test_feature_columns_exclude_the_target(seeded_db):
    frame = _build(seeded_db)
    columns = feature_columns(frame)
    for leak in ("home_goals", "away_goals", "result", "match_id"):
        assert leak not in columns


def test_form_drops_matches_older_than_the_window():
    state = TeamState()
    today = dt.date(2026, 9, 12)
    for age, goals in ((300, 5), (200, 4), (10, 0), (3, 0)):
        state.record(
            MatchOutcome(
                date=today - dt.timedelta(days=age),
                competition="premier_league",
                is_home=True,
                goals_for=goals,
                goals_against=0,
                shots_for=None,
                shots_against=None,
                shots_on_target_for=None,
                shots_on_target_against=None,
                opponent="Someone",
                opponent_elo=1500.0,
            ),
            is_league=True,
        )
    recent = state.form(6, today=today, max_age_days=120)
    assert recent["games"] == 2
    assert recent["goals_for"] == 0.0


def test_elo_is_zero_sum_and_rewards_the_winner():
    table = EloTable(FeatureConfig())
    before_home = table.rating("Alpha")
    before_away = table.rating("Bravo")
    home_change, away_change = table.update("Alpha", "Bravo", 3, 0)
    assert home_change > 0 and away_change == -home_change
    assert table.rating("Alpha") > before_home
    assert table.rating("Bravo") < before_away


def test_elo_regresses_between_seasons():
    table = EloTable(FeatureConfig())
    table.ratings["Alpha"] = 1900.0
    table._last_season["Alpha"] = "2025-26"
    table.start_season("Alpha", "2026-27", "premier_league")
    assert 1500.0 < table.ratings["Alpha"] < 1900.0


def test_promoted_clubs_start_below_the_top_flight():
    table = EloTable(FeatureConfig())
    assert table.rating("Newcomer", "championship") < table.rating("Established", "premier_league")
