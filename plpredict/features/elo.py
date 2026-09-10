"""Cross-competition Elo ratings.

Elo is the backbone of the feature set for one specific reason: it is
the only team-strength measure here that survives promotion. A club
coming up from the Championship has no Premier League form, no Premier
League table position and no head-to-head record, but it does have two
seasons of results against opposition whose strength is already known.
Rating every English competition on one scale lets that information
carry across the divide instead of being thrown away.

Three departures from textbook Elo, each earning its place:

* **Margin of victory.** A 4-0 win says more than a 1-0 win, so the
  update is scaled by goal difference (the scaling used in most public
  football Elo implementations).
* **Division offsets.** A club's first-ever rating starts below 1500 if
  it appears in a lower division, so a League One side is not treated as
  the equal of a Premier League one before they have met.
* **Between-season regression.** Squads turn over every summer, so
  ratings are pulled part-way back toward the mean between seasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plpredict.config import FeatureConfig


@dataclass
class EloTable:
    """Mutable Elo state, advanced one match at a time in date order."""

    config: FeatureConfig
    ratings: dict[str, float] = field(default_factory=dict)
    _last_season: dict[str, str] = field(default_factory=dict)

    def rating(self, team: str, competition: str = "premier_league") -> float:
        if team not in self.ratings:
            offset = self.config.elo_division_offsets.get(competition, -250.0)
            self.ratings[team] = self.config.elo_initial + offset
        return self.ratings[team]

    def start_season(self, team: str, season: str, competition: str) -> None:
        """Regress a team's rating on its first match of a new season."""
        if self._last_season.get(team) == season:
            return
        if team in self.ratings:
            pull = self.config.elo_season_regression
            self.ratings[team] = (
                (1 - pull) * self.ratings[team] + pull * self.config.elo_initial
            )
        else:
            self.rating(team, competition)
        self._last_season[team] = season

    def expected_home_score(self, home: str, away: str) -> float:
        """Probability-scale expectation for the home side, 0..1."""
        difference = (
            self.rating(home) + self.config.elo_home_advantage - self.rating(away)
        )
        return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))

    @staticmethod
    def _margin_multiplier(goal_difference: int, rating_difference: float) -> float:
        """Damp the reward for thrashing a much weaker opponent."""
        margin = abs(goal_difference)
        if margin < 2:
            return 1.0
        if margin == 2:
            return 1.5
        return (11.0 + margin) / 8.0

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        weight: float = 1.0,
    ) -> tuple[float, float]:
        """Apply one result and return the two rating changes."""
        expected = self.expected_home_score(home, away)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals < away_goals:
            actual = 0.0
        else:
            actual = 0.5

        rating_difference = (
            self.rating(home) + self.config.elo_home_advantage - self.rating(away)
        )
        multiplier = self._margin_multiplier(home_goals - away_goals, rating_difference)
        change = self.config.elo_k * multiplier * weight * (actual - expected)

        self.ratings[home] = self.rating(home) + change
        self.ratings[away] = self.rating(away) - change
        return change, -change


# Competition weights: a league match is the cleanest signal, cup ties are
# discounted because both sides may be rotating heavily.
COMPETITION_WEIGHTS = {
    "premier_league": 1.0,
    "championship": 1.0,
    "league_one": 1.0,
    "fa_cup": 0.6,
    "efl_cup": 0.5,
}
