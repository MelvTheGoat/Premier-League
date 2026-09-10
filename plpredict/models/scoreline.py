"""Scoreline model: a Dixon-Coles bivariate Poisson goal model.

Exact scores are a different problem to outcomes and deserve a different
tool. Goals are low-count events, so the natural object to model is each
side's *goal rate*, not a label. Every club gets an attack strength and
a defence strength; the home side's expected goals is its attack times
the opponent's defence times a shared home-advantage term, and the same
in reverse for the away side. A score matrix follows from those two
rates, and from it come an expected score, a most-likely score and a set
of outcome probabilities.

Two corrections to the plain independent-Poisson model, both from Dixon
and Coles (1997):

* **Low-score dependence.** Independent Poisson under-predicts 0-0 and
  1-1 and over-predicts 1-0 and 0-1. A single parameter ``rho`` adjusts
  the four cells where both sides score at most one.
* **Time decay.** Older matches carry exponentially less weight, so the
  ratings track a squad as it changes rather than averaging over an era.

Two additions of our own:

* **Second-tier matches are fitted too**, with their own scoring-level
  offset. A promoted club has no Premier League record to speak of, and
  fitting it on those few matches alone produces absurd ratings — a club
  with one goal in three games looks incapable of scoring. Including the
  Championship, where the clubs that came up and went down the other way
  tie the two scales together, gives every promoted side a rating built
  on a full season of evidence.
* **Ridge shrinkage** toward league-average strength, which keeps a club
  with a handful of matches from acquiring an extreme rating on the
  strength of one good afternoon.

Exact scorelines are inherently much less predictable than outcomes —
even a well-specified model gets a minority of them exactly right — so
the platform presents them alongside actual scores without marking them
right or wrong.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from plpredict.config import ScorelineModelConfig

_REFERENCE_COMPETITION = "premier_league"


def dixon_coles_correction(
    home_goals: np.ndarray | int,
    away_goals: np.ndarray | int,
    home_rate: np.ndarray | float,
    away_rate: np.ndarray | float,
    rho: float,
) -> np.ndarray:
    """The tau adjustment applied to the four low-score cells."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    correction = np.ones(np.broadcast(home_goals, away_goals, home_rate, away_rate).shape)

    both_zero = (home_goals == 0) & (away_goals == 0)
    home_one = (home_goals == 1) & (away_goals == 0)
    away_one = (home_goals == 0) & (away_goals == 1)
    both_one = (home_goals == 1) & (away_goals == 1)

    correction = np.where(both_zero, 1.0 - home_rate * away_rate * rho, correction)
    correction = np.where(home_one, 1.0 + away_rate * rho, correction)
    correction = np.where(away_one, 1.0 + home_rate * rho, correction)
    correction = np.where(both_one, 1.0 - rho, correction)
    return np.clip(correction, 1e-9, None)


@dataclass
class ScorelineModel:
    """Fitted attack/defence strengths plus home advantage and rho."""

    config: ScorelineModelConfig
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.25
    rho: float = -0.05
    baseline: float = 0.0
    competition_offsets: dict[str, float] = field(default_factory=dict)
    n_training_matches: int = 0
    fitted_through: str | None = None

    # -- fitting ---------------------------------------------------------

    def fit(self, frame: pd.DataFrame, as_of: dt.date) -> ScorelineModel:
        played = frame[frame["home_goals"].notna() & frame["away_goals"].notna()].copy()
        played["match_date"] = pd.to_datetime(played["match_date"]).dt.date
        # Matches too old to carry meaningful weight are dropped outright:
        # they cost fitting time and, for a club that has not played in
        # the league since, leave a rating standing on decade-old form.
        cutoff = as_of - dt.timedelta(days=self.config.max_history_days)
        played = played[played["match_date"] >= cutoff]
        if played.empty:
            raise ValueError("No played matches to fit the scoreline model on")

        if "competition" not in played:
            played["competition"] = "premier_league"

        age_days = np.array([(as_of - date).days for date in played["match_date"]], dtype=float)
        weights = np.exp(-self.config.time_decay * np.clip(age_days, 0, None))

        self.teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        index = {team: position for position, team in enumerate(self.teams)}
        count = len(self.teams)

        competitions = sorted(set(played["competition"]))
        # The target competition is the reference level, so its offset is
        # fixed at zero and the others are estimated relative to it.
        other_competitions = [name for name in competitions if name != _REFERENCE_COMPETITION]
        competition_index = {name: position for position, name in enumerate(other_competitions)}
        competition_code = np.array(
            [competition_index.get(name, -1) for name in played["competition"]]
        )

        home_index = played["home_team"].map(index).to_numpy()
        away_index = played["away_team"].map(index).to_numpy()
        home_goals = played["home_goals"].to_numpy(dtype=float)
        away_goals = played["away_goals"].to_numpy(dtype=float)

        mean_goals = float(np.average(np.r_[home_goals, away_goals], weights=np.r_[weights, weights]))
        self.baseline = float(np.log(max(mean_goals, 0.2)))

        offset_count = len(other_competitions)
        start = np.concatenate(
            [np.zeros(count), np.zeros(count), [0.25], [-0.05], np.zeros(offset_count)]
        )
        ridge = self.config.ridge

        def negative_log_likelihood(parameters: np.ndarray) -> float:
            attack = parameters[:count]
            defence = parameters[count : 2 * count]
            advantage = parameters[2 * count]
            rho = np.clip(parameters[2 * count + 1], -0.35, 0.35)
            offsets = parameters[2 * count + 2 :]
            level = (
                np.where(competition_code >= 0, offsets[np.clip(competition_code, 0, None)], 0.0)
                if offset_count
                else 0.0
            )

            home_rate = np.exp(
                self.baseline + level + attack[home_index] - defence[away_index] + advantage
            )
            away_rate = np.exp(self.baseline + level + attack[away_index] - defence[home_index])
            home_rate = np.clip(home_rate, 1e-6, 15.0)
            away_rate = np.clip(away_rate, 1e-6, 15.0)

            log_likelihood = (
                poisson.logpmf(home_goals, home_rate)
                + poisson.logpmf(away_goals, away_rate)
                + np.log(
                    dixon_coles_correction(home_goals, away_goals, home_rate, away_rate, rho)
                )
            )
            penalty = ridge * (np.sum(attack**2) + np.sum(defence**2))
            # Attack strengths are only identified up to a constant, so
            # they are pinned to sum to zero.
            penalty += 100.0 * (attack.mean() ** 2 + defence.mean() ** 2)
            return -float(np.sum(weights * log_likelihood)) + penalty

        result = minimize(
            negative_log_likelihood,
            start,
            method="L-BFGS-B",
            options={"maxiter": 400, "maxfun": 60000},
        )
        parameters = result.x
        self.attack = {team: float(parameters[index[team]]) for team in self.teams}
        self.defence = {team: float(parameters[count + index[team]]) for team in self.teams}
        self.home_advantage = float(parameters[2 * count])
        self.rho = float(np.clip(parameters[2 * count + 1], -0.35, 0.35))
        offsets = parameters[2 * count + 2 :]
        self.competition_offsets = {_REFERENCE_COMPETITION: 0.0}
        self.competition_offsets.update(
            {name: float(offsets[position]) for name, position in competition_index.items()}
        )
        self.n_training_matches = len(played)
        self.fitted_through = as_of.isoformat()
        return self

    # -- prediction ------------------------------------------------------

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Goal rates for one fixture, with a neutral fallback for a new club."""
        attack_home = self.attack.get(home_team, 0.0)
        attack_away = self.attack.get(away_team, 0.0)
        defence_home = self.defence.get(home_team, 0.0)
        defence_away = self.defence.get(away_team, 0.0)
        home_rate = float(
            np.exp(self.baseline + attack_home - defence_away + self.home_advantage)
        )
        away_rate = float(np.exp(self.baseline + attack_away - defence_home))
        return float(np.clip(home_rate, 0.05, 8.0)), float(np.clip(away_rate, 0.05, 8.0))

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Probability of every scoreline up to ``max_goals`` each."""
        home_rate, away_rate = self.expected_goals(home_team, away_team)
        size = self.config.max_goals + 1
        goals = np.arange(size)
        home_probabilities = poisson.pmf(goals, home_rate)
        away_probabilities = poisson.pmf(goals, away_rate)
        matrix = np.outer(home_probabilities, away_probabilities)

        grid_home, grid_away = np.meshgrid(goals, goals, indexing="ij")
        matrix *= dixon_coles_correction(grid_home, grid_away, home_rate, away_rate, self.rho)
        return matrix / matrix.sum()

    @staticmethod
    def _outcome_mask(matrix: np.ndarray, outcome: str) -> np.ndarray:
        size = matrix.shape[0]
        grid_home, grid_away = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        if outcome == "H":
            return grid_home > grid_away
        if outcome == "A":
            return grid_home < grid_away
        return grid_home == grid_away

    def predict(
        self, home_team: str, away_team: str, given_outcome: str | None = None
    ) -> dict[str, float | int]:
        """Expected goals, a scoreline, and the goal model's own outcome view.

        When ``given_outcome`` is supplied the reported scoreline is the
        most likely one *consistent with that outcome* rather than the
        most likely scoreline overall. The two are often different, and
        for a good reason: a home win's probability is spread across
        1-0, 2-0, 2-1 and the rest, so 1-1 can be the single most likely
        score even when a home win is comfortably the most likely result.
        Showing "home win, 1-1" would just look broken, so the outcome
        model picks the result and the goal model picks the likeliest
        score within it. The unconditional expected goals are reported
        alongside and are unaffected.
        """
        matrix = self.score_matrix(home_team, away_team)
        goals = np.arange(matrix.shape[0])

        home_expected = float((matrix.sum(axis=1) * goals).sum())
        away_expected = float((matrix.sum(axis=0) * goals).sum())

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        selectable = matrix
        if given_outcome in ("H", "D", "A"):
            selectable = np.where(self._outcome_mask(matrix, given_outcome), matrix, -1.0)
        best = np.unravel_index(int(selectable.argmax()), matrix.shape)

        return {
            "exp_home_goals": round(home_expected, 3),
            "exp_away_goals": round(away_expected, 3),
            "pred_home_goals": int(best[0]),
            "pred_away_goals": int(best[1]),
            "scoreline_prob": float(matrix[best]),
            "poisson_p_home": home_win,
            "poisson_p_draw": draw,
            "poisson_p_away": away_win,
        }

    def predict_frame(
        self, frame: pd.DataFrame, given_outcomes: pd.Series | None = None
    ) -> pd.DataFrame:
        outcomes = (
            given_outcomes.reindex(frame.index)
            if given_outcomes is not None
            else pd.Series(None, index=frame.index, dtype=object)
        )
        predictions = [
            self.predict(
                record["home_team"], record["away_team"], given_outcome=outcomes.loc[index]
            )
            for index, record in frame.iterrows()
        ]
        result = pd.DataFrame(predictions)
        result.index = frame.index
        return result

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path: Path) -> ScorelineModel:
        with open(path, "rb") as handle:
            return pickle.load(handle)
