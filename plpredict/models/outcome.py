"""Match-outcome model: probabilities for home win, draw and away win.

A small ensemble rather than a single learner, because the two halves
fail in different places:

* **Gradient boosting** (LightGBM) finds the interactions that make the
  contextual features worth collecting at all — that a congested
  fixture matters more for a squad with key players missing, that a new
  manager's effect depends on where the club sits in the table. It needs
  data to do that, and it can be over-confident on the thin evidence
  available in the opening weeks of a season.
* **Multinomial logistic regression** on a compact core of strength and
  form features is far more stable and better calibrated early, and it
  cannot invent an interaction it has not got the data for.

Serving the blend of the two is materially better calibrated across a
whole season than either alone, which matters here because the platform
publishes probabilities, not just a pick.

Draws are the hard class: they are never the most likely outcome for a
lopsided fixture and only rarely for an even one, so a model optimised
for accuracy alone learns to stop predicting them. Training therefore
optimises log loss and the reported pick is the argmax of the blended
probabilities, with the probabilities themselves shown in the UI.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from plpredict.config import OutcomeModelConfig

CLASSES = ("H", "D", "A")

# The compact set the linear half of the ensemble is trained on. These
# are the features that are always populated (they derive from results
# alone), which is what makes the linear model dependable in the weeks
# when the richer context is still sparse.
CORE_FEATURES = (
    "elo_expected_home",
    "diff_elo",
    "diff_form6_ppg",
    "diff_form6_goal_difference",
    "diff_form10_ppg",
    "diff_venue_ppg",
    "diff_previous_season_ppg",
    "diff_table_ppg",
    "home_venue_ppg",
    "away_venue_ppg",
    "h2h_home_ppg",
    "season_progress",
)


@dataclass
class OutcomeModel:
    """Fitted ensemble plus everything needed to reproduce its inputs."""

    config: OutcomeModelConfig
    columns: list[str] = field(default_factory=list)
    core_columns: list[str] = field(default_factory=list)
    booster: Any = None
    linear: Pipeline | None = None
    n_training_matches: int = 0
    chosen_estimators: int = 0

    # -- fitting ---------------------------------------------------------

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> OutcomeModel:
        import lightgbm as lgb

        labelled = frame[frame["result"].notna()]
        if labelled.empty:
            raise ValueError("No labelled matches to train on")
        if not columns:
            raise ValueError(
                "No usable feature columns: every candidate was either absent "
                "from the target gameweek or too sparse in the training data"
            )

        self.columns = list(columns)
        self.core_columns = [name for name in CORE_FEATURES if name in columns]
        self.n_training_matches = len(labelled)

        # Chronological order matters: the early-stopping probe below
        # validates on the most recent matches, not a random sample.
        labelled = labelled.sort_values("match_date")
        features = labelled[self.columns].astype(float)
        target = labelled["result"].map({name: index for index, name in enumerate(CLASSES)})

        settings = self.config
        rounds = self._choose_rounds(features, target)
        self.chosen_estimators = rounds
        self.booster = lgb.LGBMClassifier(
            **self._booster_params(n_estimators=rounds)
        )
        self.booster.fit(features, target)

        if self.core_columns:
            self.linear = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        # Multinomial (softmax) is scikit-learn's default
                        # for a multiclass target and is what we want here:
                        # the three outcomes are one choice, not three
                        # independent yes/no questions.
                        LogisticRegression(max_iter=2000, C=0.5),
                    ),
                ]
            )
            self.linear.fit(labelled[self.core_columns].astype(float), target)
        return self

    def _booster_params(self, n_estimators: int) -> dict[str, Any]:
        settings = self.config
        return {
            "objective": "multiclass",
            "num_class": len(CLASSES),
            "learning_rate": settings.learning_rate,
            "num_leaves": settings.num_leaves,
            "min_child_samples": settings.min_child_samples,
            "n_estimators": n_estimators,
            "subsample": settings.subsample,
            "subsample_freq": settings.subsample_freq,
            "colsample_bytree": settings.colsample_bytree,
            "reg_lambda": settings.reg_lambda,
            "random_state": settings.random_state,
            "verbose": -1,
        }

    def _choose_rounds(self, features: pd.DataFrame, target: pd.Series) -> int:
        """Pick the number of boosting rounds by early stopping.

        A probe is trained on all but the most recent slice of the
        training set and stopped when log loss on that slice stops
        improving. The count it settles on is then scaled up in
        proportion to the data the probe did not see, and the model that
        actually gets served is refitted on everything — the last few
        weeks of results are the most relevant matches in the set and
        should not be spent on validation.
        """
        import lightgbm as lgb

        settings = self.config
        cut = int(len(features) * (1 - settings.validation_fraction))
        if cut < 200 or len(features) - cut < 60:
            # Too little data to stop on; fall back to a modest fixed
            # count rather than the cap, which would badly overfit.
            return min(settings.max_estimators, 150)

        probe = lgb.LGBMClassifier(**self._booster_params(settings.max_estimators))
        probe.fit(
            features.iloc[:cut],
            target.iloc[:cut],
            eval_set=[(features.iloc[cut:], target.iloc[cut:])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(settings.early_stopping_rounds, verbose=False)],
        )
        best = probe.best_iteration_ or settings.max_estimators
        scaled = int(round(best / (1 - settings.validation_fraction)))
        return max(settings.min_estimators, min(scaled, settings.max_estimators))

    # -- prediction ------------------------------------------------------

    @staticmethod
    def _to_full_classes(probabilities: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """Widen a fitted model's output back to all three outcomes.

        scikit-learn and LightGBM both emit one column per class *present
        in the training data*. Over a full Premier League history all
        three always are, but a short or unusual training window can
        contain no draws, and silently returning a two-column array would
        break the blend. Absent classes come back as zero probability,
        which is what the training data actually said.
        """
        if probabilities.shape[1] == len(CLASSES):
            return probabilities
        widened = np.zeros((probabilities.shape[0], len(CLASSES)))
        for position, label in enumerate(seen):
            widened[:, int(label)] = probabilities[:, position]
        return widened

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been fitted")
        boosted = self._to_full_classes(
            self.booster.predict_proba(frame[self.columns].astype(float)),
            self.booster.classes_,
        )
        if self.linear is None:
            return boosted
        linear = self._to_full_classes(
            self.linear.predict_proba(frame[self.core_columns].astype(float)),
            self.linear.named_steps["model"].classes_,
        )
        weight = self.config.blend_weight
        blended = weight * boosted + (1 - weight) * linear
        return blended / blended.sum(axis=1, keepdims=True)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(frame)
        result = pd.DataFrame(probabilities, columns=["p_home", "p_draw", "p_away"])
        result.index = frame.index
        result["predicted_outcome"] = [CLASSES[index] for index in probabilities.argmax(axis=1)]
        result["confidence"] = probabilities.max(axis=1)
        return result

    def feature_importance(self, top: int = 25) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        importances = zip(self.columns, self.booster.feature_importances_)
        ranked = sorted(importances, key=lambda item: item[1], reverse=True)
        return [(name, float(value)) for name, value in ranked[:top]]

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path: Path) -> OutcomeModel:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def usable_columns(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    candidates: list[str],
    min_coverage: int = 50,
    min_target_coverage: float = 0.5,
) -> list[str]:
    """Keep only features that exist on both sides of the split.

    Some sources — match statistics above all — lag the live season by
    weeks. A column that is missing for the gameweek being predicted is
    dropped from that run rather than served as a missing value, so the
    model is never asked to lean on evidence it will not have. The
    moment the source catches up, the column returns on the next retrain
    without any code change.

    A column has to be populated for at least ``min_target_coverage`` of
    the fixtures being predicted, not merely one of them. A single
    surviving value is not enough to justify a feature the model may
    have learned to depend on across thousands of training matches.
    """
    keep = []
    for column in candidates:
        if column not in train or column not in predict:
            continue
        if train[column].notna().sum() < min_coverage:
            continue
        if predict.empty:
            keep.append(column)
            continue
        if predict[column].notna().mean() >= min_target_coverage:
            keep.append(column)
    return keep
