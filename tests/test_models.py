"""Both models, checked for the properties the platform depends on."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from plpredict import db
from plpredict.config import OUTCOME_MODEL, SCORELINE_MODEL
from plpredict.features.build import FeatureBuilder, feature_columns
from plpredict.models.outcome import CLASSES, OutcomeModel, usable_columns
from plpredict.models.scoreline import ScorelineModel, dixon_coles_correction


@pytest.fixture(scope="module")
def features(seeded_db):
    with db.connect(seeded_db) as conn:
        return FeatureBuilder(conn).build()


@pytest.fixture(scope="module")
def matches(seeded_db):
    with db.connect(seeded_db) as conn:
        return pd.read_sql(
            "SELECT season, competition, match_date, home_team, away_team, "
            "home_goals, away_goals FROM matches WHERE status = 'played'",
            conn,
        )


def test_outcome_probabilities_are_a_distribution(features):
    labelled = features[features["result"].notna()]
    upcoming = features[features["result"].isna()]
    columns = usable_columns(
        labelled, upcoming, feature_columns(features), min_coverage=20
    )
    model = OutcomeModel(OUTCOME_MODEL).fit(labelled, columns)

    probabilities = model.predict_proba(upcoming)
    assert probabilities.shape == (len(upcoming), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()

    predictions = model.predict_frame(upcoming)
    assert set(predictions["predicted_outcome"]) <= set(CLASSES)
    # The reported confidence is the probability of the pick.
    assert np.allclose(predictions["confidence"], probabilities.max(axis=1))


def test_usable_columns_drops_features_missing_from_the_target(features):
    labelled = features[features["result"].notna()].copy()
    upcoming = features[features["result"].isna()].copy()
    candidates = feature_columns(features)

    lagging = "diff_form6_ppg"
    assert lagging in candidates
    upcoming[lagging] = np.nan
    assert lagging not in usable_columns(
        labelled, upcoming, candidates, min_coverage=20
    )


def test_scoreline_model_produces_a_proper_score_distribution(matches):
    model = ScorelineModel(SCORELINE_MODEL).fit(matches, dt.date(2026, 10, 1))
    matrix = model.score_matrix("Alpha", "Bravo")
    assert matrix.shape == (SCORELINE_MODEL.max_goals + 1,) * 2
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()

    prediction = model.predict("Alpha", "Bravo")
    total = (
        prediction["poisson_p_home"]
        + prediction["poisson_p_draw"]
        + prediction["poisson_p_away"]
    )
    assert total == pytest.approx(1.0)
    assert prediction["exp_home_goals"] > 0


def test_home_advantage_is_learned_from_data_that_has_it():
    """Two evenly matched clubs, home sides scoring more: rho aside, the
    only parameter that can explain it is home advantage."""
    rows = []
    for round_number in range(60):
        for home, away in (("Alpha", "Bravo"), ("Bravo", "Alpha")):
            rows.append(
                {
                    "season": "2025-26",
                    "competition": "premier_league",
                    "match_date": (dt.date(2025, 8, 10) + dt.timedelta(days=round_number)).isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 2,
                    "away_goals": 1,
                }
            )
    model = ScorelineModel(SCORELINE_MODEL).fit(pd.DataFrame(rows), dt.date(2026, 1, 1))
    assert model.home_advantage > 0

    home_rate, away_rate = model.expected_goals("Alpha", "Bravo")
    assert home_rate > away_rate


def test_conditional_scoreline_matches_the_given_outcome(matches):
    model = ScorelineModel(SCORELINE_MODEL).fit(matches, dt.date(2026, 10, 1))
    for outcome, comparison in (
        ("H", lambda home, away: home > away),
        ("D", lambda home, away: home == away),
        ("A", lambda home, away: home < away),
    ):
        prediction = model.predict("Alpha", "Bravo", given_outcome=outcome)
        assert comparison(prediction["pred_home_goals"], prediction["pred_away_goals"])


def test_dixon_coles_correction_only_touches_low_scores():
    rho = -0.1
    assert dixon_coles_correction(3, 2, 1.4, 1.1, rho) == pytest.approx(1.0)
    assert dixon_coles_correction(0, 0, 1.4, 1.1, rho) != pytest.approx(1.0)
    # A negative rho lifts 0-0 and 1-1 and depresses 1-0 and 0-1, which is
    # the whole point of the adjustment.
    assert dixon_coles_correction(0, 0, 1.4, 1.1, rho) > 1.0
    assert dixon_coles_correction(1, 1, 1.4, 1.1, rho) > 1.0
    assert dixon_coles_correction(1, 0, 1.4, 1.1, rho) < 1.0
