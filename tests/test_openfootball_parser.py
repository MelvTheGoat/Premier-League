"""The parser is load-bearing: everything downstream trusts its output."""

from __future__ import annotations

import datetime as dt

from plpredict.data.sources.openfootball import SeasonFileParser, _normalise_result_notation

MATCHDAY_FORMAT = """
= English Premier League 2024/25

# Date  Fri Aug 16 2024 - Sun May 25 2025

▪ Matchday 1
  Fri Aug 16 2024
    20:00  Manchester United FC    v Fulham FC                1-0 (0-0)
  Sat Aug 17
    15:00  Arsenal FC              v Wolverhampton Wanderers FC  2-0 (1-0)
           Everton FC              v Brighton & Hove Albion FC  0-3 (0-1)

▪ Matchday 2
  Sat Aug 24
    15:00  Brentford FC            v Crystal Palace FC
"""

SCORE_IN_MIDDLE_FORMAT = """
= England | Premier League 2025/26

▪ Regular Season - 1
Fri Aug 15 2025
  19:00   Liverpool  4-2 (1-0)  Bournemouth
                  (Hugo EKITIKE 37', Cody GAKPO 49';
                   Antoine SEMENYO 64', 76')
Sat Aug 16
  12:30   Aston Villa  0-0 (0-0)  Newcastle United
"""

YEAR_ROLLOVER = """
▪ Matchday 20
  Sat Dec 28 2024
    15:00  Arsenal FC  v Chelsea FC  1-1 (0-1)
  Wed Jan 1
    15:00  Everton FC  v Fulham FC   2-0 (1-0)
"""


def test_parses_matchday_format():
    matches = SeasonFileParser("2024-25", "premier_league").parse(MATCHDAY_FORMAT)
    assert len(matches) == 4

    first = matches[0]
    assert (first.home_team, first.away_team) == ("Manchester United FC", "Fulham FC")
    assert (first.home_goals, first.away_goals) == (1, 0)
    assert (first.ht_home_goals, first.ht_away_goals) == (0, 0)
    assert first.date == dt.date(2024, 8, 16)
    assert first.kickoff == "20:00"
    assert first.matchday == 1

    # The date carries over to the following lines, the kick-off time too.
    assert matches[2].date == dt.date(2024, 8, 17)
    assert matches[2].kickoff == "15:00"


def test_unplayed_fixture_has_no_score():
    matches = SeasonFileParser("2024-25", "premier_league").parse(MATCHDAY_FORMAT)
    upcoming = matches[-1]
    assert upcoming.matchday == 2
    assert upcoming.home_goals is None
    assert not upcoming.is_played


def test_parses_score_in_middle_format_and_skips_scorer_lines():
    matches = SeasonFileParser("2025-26", "premier_league").parse(SCORE_IN_MIDDLE_FORMAT)
    assert len(matches) == 2
    assert matches[0].home_team == "Liverpool"
    assert matches[0].away_team == "Bournemouth"
    assert (matches[0].home_goals, matches[0].away_goals) == (4, 2)
    assert matches[1].home_goals == 0


def test_year_rolls_over_at_new_year():
    matches = SeasonFileParser("2024-25", "premier_league").parse(YEAR_ROLLOVER)
    assert matches[0].date == dt.date(2024, 12, 28)
    assert matches[1].date == dt.date(2025, 1, 1)


def test_knockout_notation_reports_the_regulation_score():
    # The bracketed pairs are the 90-minute and half-time scores; the
    # leading pair is the shoot-out, which is not a football score.
    line = "Nottingham Forest  v Bury FC  10-9 pen. (1-1, 0-1)"
    match = SeasonFileParser("2024-25", "efl_cup")._parse_fixture(line)
    assert (match.home_goals, match.away_goals) == (1, 1)
    assert (match.ht_home_goals, match.ht_away_goals) == (0, 1)

    line = "Crawley Town  v Southend United  2-6 a.e.t. (2-2, 0-2)"
    match = SeasonFileParser("2024-25", "fa_cup")._parse_fixture(line)
    assert (match.home_goals, match.away_goals) == (2, 2)


def test_abandoned_match_is_dropped():
    _, keep = _normalise_result_notation("Scunthorpe United v Wealdstone FC  [abandoned]")
    assert keep is False
    assert SeasonFileParser("2024-25", "fa_cup")._parse_fixture(
        "Scunthorpe United v Wealdstone FC  [abandoned]"
    ) is None
