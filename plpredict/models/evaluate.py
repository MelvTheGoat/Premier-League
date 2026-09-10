"""Walk-forward evaluation.

A single train/test split would flatter this model, because a random
split lets it learn from matches that had not happened yet. The only
honest test is the one that mirrors how the platform actually runs:
predict gameweek N knowing only gameweeks 1..N-1, then move on. That is
what ``walk_forward`` does, gameweek by gameweek, across whole seasons.

Accuracy alone is a poor score for a three-outcome problem with an
unpredictable middle class, so the report also carries:

``log_loss``    rewards being right *and* appropriately uncertain; the
                headline number for a model that publishes probabilities
``brier``       the same idea, less punishing about confident misses
``vs_home``     accuracy against always predicting a home win
``vs_elo``      log loss against an Elo-only baseline, which is what the
                contextual features have to beat to justify themselves
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from plpredict import config
from plpredict.features.build import feature_columns
from plpredict.models.outcome import CLASSES, OutcomeModel, usable_columns
from plpredict.models.scoreline import ScorelineModel

_EPSILON = 1e-12


def log_loss(probabilities: np.ndarray, actual: np.ndarray) -> float:
    picked = probabilities[np.arange(len(actual)), actual]
    return float(-np.mean(np.log(np.clip(picked, _EPSILON, 1.0))))


def brier_score(probabilities: np.ndarray, actual: np.ndarray) -> float:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


@dataclass
class Evaluation:
    n_matches: int
    accuracy: float
    log_loss: float
    brier: float
    baseline_home_accuracy: float
    baseline_elo_log_loss: float
    exact_scoreline_rate: float
    mean_absolute_goal_error: float
    by_season: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: list[dict[str, float]] = field(default_factory=list)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.n_matches} matches | accuracy {self.accuracy:.1%} "
            f"(always-home {self.baseline_home_accuracy:.1%}) | "
            f"log loss {self.log_loss:.4f} (Elo-only {self.baseline_elo_log_loss:.4f}) | "
            f"Brier {self.brier:.4f} | exact score {self.exact_scoreline_rate:.1%} | "
            f"goals MAE {self.mean_absolute_goal_error:.2f}"
        )


def _elo_baseline_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """Elo expectation spread over three outcomes, as a reference point.

    Elo gives one number, the home side's expected score. Draws are
    carved out of it with a fixed share that falls away as the fixture
    becomes more lopsided, which is roughly how draw frequency behaves.
    """
    expectation = frame["elo_expected_home"].to_numpy(dtype=float)
    imbalance = np.abs(expectation - 0.5) * 2.0
    draw = np.clip(0.30 - 0.22 * imbalance, 0.08, 0.32)
    remaining = 1.0 - draw
    home = remaining * expectation
    away = remaining * (1.0 - expectation)
    return np.clip(np.column_stack([home, draw, away]), _EPSILON, 1.0)


def calibration_table(
    probabilities: np.ndarray, actual: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Predicted probability against observed frequency, pooled over classes."""
    flat_probability = probabilities.reshape(-1)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(actual)), actual] = 1.0
    flat_actual = onehot.reshape(-1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (flat_probability >= low) & (flat_probability < high)
        if not selected.any():
            continue
        rows.append(
            {
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "predicted": round(float(flat_probability[selected].mean()), 4),
                "observed": round(float(flat_actual[selected].mean()), 4),
                "n": int(selected.sum()),
            }
        )
    return rows


def walk_forward(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    seasons: list[str],
    *,
    include_scoreline: bool = True,
    min_matchday: int = 1,
    verbose: bool = False,
) -> Evaluation:
    """Replay the platform's own procedure over a set of seasons."""
    candidates = feature_columns(features)
    collected: list[dict[str, Any]] = []

    for season in seasons:
        season_rows = features[features["season"] == season]
        matchdays = sorted(
            int(value)
            for value in season_rows["matchday"].dropna().unique()
            if value >= min_matchday
        )
        for matchday in matchdays:
            target = season_rows[season_rows["matchday"] == matchday]
            target = target[target["result"].notna()]
            if target.empty:
                continue
            cutoff = min(dt.date.fromisoformat(value) for value in target["match_date"])

            history = features[
                (pd.to_datetime(features["match_date"]).dt.date < cutoff)
                & features["result"].notna()
            ]
            if len(history) < 500:
                continue

            columns = usable_columns(history, target, candidates)
            model = OutcomeModel(config.OUTCOME_MODEL).fit(history, columns)
            probabilities = model.predict_proba(target)
            predicted = probabilities.argmax(axis=1)

            scores = None
            if include_scoreline:
                played_before = matches[
                    pd.to_datetime(matches["match_date"]).dt.date < cutoff
                ]
                goal_model = ScorelineModel(config.SCORELINE_MODEL).fit(played_before, cutoff)
                scores = goal_model.predict_frame(
                    target,
                    given_outcomes=pd.Series(
                        [CLASSES[index] for index in predicted], index=target.index
                    ),
                )

            for position, (index, record) in enumerate(target.iterrows()):
                entry = {
                    "season": season,
                    "matchday": matchday,
                    "actual": CLASSES.index(record["result"]),
                    "predicted": int(predicted[position]),
                    "p_home": probabilities[position, 0],
                    "p_draw": probabilities[position, 1],
                    "p_away": probabilities[position, 2],
                    "elo_expected_home": record["elo_expected_home"],
                    "home_goals": record["home_goals"],
                    "away_goals": record["away_goals"],
                }
                if scores is not None:
                    entry["pred_home_goals"] = scores.loc[index, "pred_home_goals"]
                    entry["pred_away_goals"] = scores.loc[index, "pred_away_goals"]
                    entry["exp_home_goals"] = scores.loc[index, "exp_home_goals"]
                    entry["exp_away_goals"] = scores.loc[index, "exp_away_goals"]
                collected.append(entry)

            if verbose:
                print(f"    {season} GW{matchday}: {len(target)} matches", flush=True)

    frame = pd.DataFrame(collected)
    if frame.empty:
        raise ValueError("Walk-forward produced no evaluated matches")

    probabilities = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=int)

    exact = float("nan")
    goal_error = float("nan")
    if "pred_home_goals" in frame:
        exact = float(
            (
                (frame["pred_home_goals"] == frame["home_goals"])
                & (frame["pred_away_goals"] == frame["away_goals"])
            ).mean()
        )
        goal_error = float(
            (
                (frame["exp_home_goals"] - frame["home_goals"]).abs()
                + (frame["exp_away_goals"] - frame["away_goals"]).abs()
            ).mean()
            / 2
        )

    by_season = {}
    for season, group in frame.groupby("season"):
        group_probabilities = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        group_actual = group["actual"].to_numpy(dtype=int)
        by_season[str(season)] = {
            "n": int(len(group)),
            "accuracy": float((group["predicted"] == group["actual"]).mean()),
            "log_loss": log_loss(group_probabilities, group_actual),
        }

    confusion: dict[str, dict[str, int]] = {
        name: {inner: 0 for inner in CLASSES} for name in CLASSES
    }
    for row in frame.itertuples():
        confusion[CLASSES[row.actual]][CLASSES[row.predicted]] += 1

    return Evaluation(
        n_matches=len(frame),
        accuracy=float((frame["predicted"] == frame["actual"]).mean()),
        log_loss=log_loss(probabilities, actual),
        brier=brier_score(probabilities, actual),
        baseline_home_accuracy=float((frame["actual"] == 0).mean()),
        baseline_elo_log_loss=log_loss(_elo_baseline_probabilities(frame), actual),
        exact_scoreline_rate=exact,
        mean_absolute_goal_error=goal_error,
        by_season=by_season,
        calibration=calibration_table(probabilities, actual),
        confusion=confusion,
    )
