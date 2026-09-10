"""Per-team rolling state, advanced strictly in date order.

Everything here answers the same question: *what was knowable about this
club the day before this match kicked off?* Keeping the answer in a
mutable object that only ever moves forward in time is what makes the
point-in-time guarantee structural rather than something to remember —
a feature physically cannot see a result that has not been fed in yet.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from dataclasses import dataclass, field

MAX_HISTORY = 20


@dataclass
class MatchOutcome:
    """One completed match from a single team's point of view."""

    date: dt.date
    competition: str
    is_home: bool
    goals_for: int
    goals_against: int
    shots_for: float | None
    shots_against: float | None
    shots_on_target_for: float | None
    shots_on_target_against: float | None
    opponent: str
    opponent_elo: float

    @property
    def points(self) -> int:
        if self.goals_for > self.goals_against:
            return 3
        return 1 if self.goals_for == self.goals_against else 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class TeamState:
    """Rolling history for one club."""

    recent: deque[MatchOutcome] = field(
        default_factory=lambda: deque(maxlen=MAX_HISTORY)
    )
    # League-only, current-season accumulators used for the table.
    season: str | None = None
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_played: int = 0
    home_points: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_played: int = 0
    away_points: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    # Previous season's league record, which is all a club has to go on
    # for the first weeks of a new one.
    previous_ppg: float | None = None
    previous_goal_difference_per_game: float | None = None
    previous_competition: str | None = None
    competition: str | None = None
    seasons_seen: int = 0
    top_flight_seasons: int = 0
    # Every fixture in any competition, for congestion.
    match_dates: deque[dt.date] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn

    def roll_season(self, season: str, competition: str) -> None:
        """Close the previous league season and open a new one.

        Seasons are rolled for every league a club plays in, not just the
        top flight. Otherwise a club that has just come up carries a
        "previous season" record from whenever it was last in the top
        division, which for some clubs is a decade ago and for all of
        them skips the season that actually got them promoted.
        """
        if self.season == season:
            return
        if self.season is not None and self.played:
            self.previous_ppg = self.points / self.played
            self.previous_goal_difference_per_game = (
                self.goals_for - self.goals_against
            ) / self.played
            self.previous_competition = self.competition
            self.seasons_seen += 1
            if self.competition == "premier_league":
                self.top_flight_seasons += 1
        self.season = season
        self.competition = competition
        self.played = self.won = self.drawn = self.lost = 0
        self.goals_for = self.goals_against = 0
        self.home_played = self.home_points = 0
        self.home_goals_for = self.home_goals_against = 0
        self.away_played = self.away_points = 0
        self.away_goals_for = self.away_goals_against = 0

    def record(self, outcome: MatchOutcome, is_league: bool) -> None:
        self.recent.append(outcome)
        self.match_dates.append(outcome.date)
        if not is_league:
            return
        self.played += 1
        self.goals_for += outcome.goals_for
        self.goals_against += outcome.goals_against
        if outcome.points == 3:
            self.won += 1
        elif outcome.points == 1:
            self.drawn += 1
        else:
            self.lost += 1
        if outcome.is_home:
            self.home_played += 1
            self.home_points += outcome.points
            self.home_goals_for += outcome.goals_for
            self.home_goals_against += outcome.goals_against
        else:
            self.away_played += 1
            self.away_points += outcome.points
            self.away_goals_for += outcome.goals_for
            self.away_goals_against += outcome.goals_against

    # -- rolling form ----------------------------------------------------

    def form(
        self, window: int, today: dt.date | None = None, max_age_days: int = 120
    ) -> dict[str, float | None]:
        """Recent form across every competition, most recent ``window`` games.

        Form deliberately spans competitions and seasons — a club's last
        six games are its last six games, whichever badge was on the
        fixture — but not indefinitely. Games older than ``max_age_days``
        are dropped, so the previous season's run-in fades out of the
        window over the opening weeks of a new one rather than being
        weighed equally with results under a rebuilt squad. In August,
        when nothing else exists, those games are the best evidence
        available and are used.
        """
        games = list(self.recent)
        if today is not None:
            games = [game for game in games if (today - game.date).days <= max_age_days]
        games = games[-window:]
        if not games:
            return {
                "ppg": None,
                "goals_for": None,
                "goals_against": None,
                "goal_difference": None,
                "win_rate": None,
                "clean_sheet_rate": None,
                "shots_for": None,
                "shots_against": None,
                "shots_on_target_for": None,
                "shots_on_target_against": None,
                "opponent_elo": None,
                "games": 0,
            }
        count = len(games)

        def mean_of(values: list[float | None]) -> float | None:
            present = [value for value in values if value is not None]
            return sum(present) / len(present) if present else None

        return {
            "ppg": sum(game.points for game in games) / count,
            "goals_for": sum(game.goals_for for game in games) / count,
            "goals_against": sum(game.goals_against for game in games) / count,
            "goal_difference": sum(game.goal_difference for game in games) / count,
            "win_rate": sum(1 for game in games if game.points == 3) / count,
            "clean_sheet_rate": sum(1 for game in games if game.goals_against == 0) / count,
            "shots_for": mean_of([game.shots_for for game in games]),
            "shots_against": mean_of([game.shots_against for game in games]),
            "shots_on_target_for": mean_of([game.shots_on_target_for for game in games]),
            "shots_on_target_against": mean_of(
                [game.shots_on_target_against for game in games]
            ),
            "opponent_elo": mean_of([game.opponent_elo for game in games]),
            "games": count,
        }

    # -- congestion ------------------------------------------------------

    def days_since_last_match(self, today: dt.date, cap: float) -> float | None:
        if not self.match_dates:
            return None
        return min((today - self.match_dates[-1]).days, cap)

    def matches_within(self, today: dt.date, days: int) -> int:
        cutoff = today - dt.timedelta(days=days)
        return sum(1 for date in self.match_dates if cutoff <= date < today)


@dataclass
class HeadToHead:
    """Meetings between two clubs, keyed by the unordered pair."""

    history: dict[tuple[str, str], deque[MatchOutcome]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=10))
    )

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def record(
        self,
        home: str,
        away: str,
        date: dt.date,
        home_goals: int,
        away_goals: int,
        competition: str,
    ) -> None:
        # Stored from the alphabetically-first club's point of view, so
        # both directions read consistently.
        first, _ = self._key(home, away)
        as_first = first == home
        self.history[self._key(home, away)].append(
            MatchOutcome(
                date=date,
                competition=competition,
                is_home=as_first,
                goals_for=home_goals if as_first else away_goals,
                goals_against=away_goals if as_first else home_goals,
                shots_for=None,
                shots_against=None,
                shots_on_target_for=None,
                shots_on_target_against=None,
                opponent=away if as_first else home,
                opponent_elo=0.0,
            )
        )

    def summary(
        self, home: str, away: str, today: dt.date, window: int = 6
    ) -> dict[str, float | None]:
        games = list(self.history.get(self._key(home, away), ()))[-window:]
        if not games:
            return {
                "h2h_games": 0,
                "h2h_home_ppg": None,
                "h2h_goal_difference": None,
                "h2h_days_since": None,
            }
        first, _ = self._key(home, away)
        home_is_first = first == home
        points = 0
        difference = 0
        for game in games:
            goals_for = game.goals_for if home_is_first else game.goals_against
            goals_against = game.goals_against if home_is_first else game.goals_for
            difference += goals_for - goals_against
            points += 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)
        return {
            "h2h_games": len(games),
            "h2h_home_ppg": points / len(games),
            "h2h_goal_difference": difference / len(games),
            "h2h_days_since": (today - games[-1].date).days,
        }


@dataclass
class LeagueTable:
    """Standings as they stood before a given match, for one league season."""

    states: dict[str, TeamState]

    def snapshot(self, teams: set[str]) -> dict[str, dict[str, float]]:
        rows = []
        for team in teams:
            state = self.states.get(team)
            if state is None:
                continue
            rows.append(
                {
                    "team": team,
                    "points": state.points,
                    "played": state.played,
                    "goal_difference": state.goals_for - state.goals_against,
                    "goals_for": state.goals_for,
                }
            )
        rows.sort(
            key=lambda row: (row["points"], row["goal_difference"], row["goals_for"]),
            reverse=True,
        )
        return {row["team"]: {**row, "position": index + 1} for index, row in enumerate(rows)}
