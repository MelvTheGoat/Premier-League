"""End-to-end: the rolling retrain, and the pages that read its output."""

from __future__ import annotations

import pandas as pd
import pytest

from plpredict import db
from plpredict.features.build import FeatureBuilder
from plpredict.pipeline.run import (
    gameweek_cutoff,
    last_completed_gameweek,
    load_match_frame,
    next_unplayed_gameweek,
    train_and_predict,
)
from plpredict.web import queries
from plpredict.web.app import create_app

SEASON = "2026-27"


@pytest.fixture(scope="module")
def predicted_db(seeded_db):
    """Run the real pipeline over the synthetic league, once for the module.

    ``min_feature_coverage`` is lowered because the synthetic league is
    far smaller than a real one; the production default guards against
    training on a column that is populated for only a handful of matches.
    """
    with db.connect(seeded_db) as conn:
        features = FeatureBuilder(conn).build()
        matches = load_match_frame(conn)
        completed = last_completed_gameweek(conn, SEASON)
        upcoming = next_unplayed_gameweek(conn, SEASON)
        for matchday in list(range(1, completed + 1)) + [upcoming]:
            train_and_predict(
                conn, features, matches, SEASON, matchday, min_feature_coverage=20
            )
    return seeded_db


def test_training_set_grows_with_each_gameweek(predicted_db):
    with db.connect(predicted_db) as conn:
        runs = conn.execute(
            "SELECT target_matchday, n_training_matches, trained_through "
            "FROM model_runs WHERE target_season = ? ORDER BY target_matchday",
            (SEASON,),
        ).fetchall()
    sizes = [row["n_training_matches"] for row in runs]
    assert len(sizes) > 2
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0], "later gameweeks must train on more matches"


def test_a_gameweek_is_never_trained_on_itself(predicted_db):
    with db.connect(predicted_db) as conn:
        features = FeatureBuilder(conn).build()
        runs = conn.execute(
            "SELECT target_matchday, trained_through FROM model_runs WHERE target_season = ?",
            (SEASON,),
        ).fetchall()
        for run in runs:
            cutoff = gameweek_cutoff(features, SEASON, run["target_matchday"])
            assert run["trained_through"] == cutoff.isoformat()
            # No match in the target gameweek kicked off before the cutoff.
            target = features[
                (features["season"] == SEASON)
                & (features["matchday"] == run["target_matchday"])
            ]
            assert (pd.to_datetime(target["match_date"]).dt.date >= cutoff).all()


def test_every_fixture_gets_exactly_one_current_prediction(predicted_db):
    with db.connect(predicted_db) as conn:
        rows = conn.execute(
            """
            SELECT COUNT(*) AS n, COUNT(DISTINCT match_id) AS distinct_matches
            FROM current_predictions WHERE season = ?
            """,
            (SEASON,),
        ).fetchone()
    assert rows["n"] == rows["distinct_matches"]


def test_predictions_are_kept_not_overwritten(predicted_db):
    """Re-predicting a gameweek adds a run; the old one stays on record."""
    with db.connect(predicted_db) as conn:
        features = FeatureBuilder(conn).build()
        matches = load_match_frame(conn)
        before = conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        train_and_predict(
            conn, features, matches, SEASON, 1, min_feature_coverage=20
        )
        after = conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        runs_for_gw1 = conn.execute(
            "SELECT COUNT(*) FROM model_runs WHERE target_season = ? AND target_matchday = 1",
            (SEASON,),
        ).fetchone()[0]
    assert after == before + 1
    assert runs_for_gw1 == 2


def test_completed_gameweeks_are_marked_right_or_wrong(predicted_db):
    with db.connect(predicted_db) as conn:
        data = queries.gameweek(conn, SEASON, 1)
    assert data["is_complete"]
    assert data["n_scored"] == len(data["matches"])
    for match in data["matches"]:
        assert match["outcome_correct"] in (True, False)
    assert 0 <= data["accuracy"] <= 1


def test_upcoming_gameweek_has_predictions_but_no_verdicts(predicted_db):
    with db.connect(predicted_db) as conn:
        upcoming = next_unplayed_gameweek(conn, SEASON)
        data = queries.gameweek(conn, SEASON, upcoming)
    assert not data["any_played"]
    assert data["n_scored"] == 0
    for match in data["matches"]:
        assert match["has_prediction"]
        assert match["outcome_correct"] is None
        assert match["pred_home_goals"] is not None
        assert sum(match["probabilities"].values()) == pytest.approx(1.0)


def test_pages_render(predicted_db):
    client = create_app(str(predicted_db)).test_client()
    assert client.get("/").status_code == 302
    assert client.get(f"/season/{SEASON}/gameweek/1").status_code == 200
    assert client.get(f"/season/{SEASON}").status_code == 200
    assert client.get("/model").status_code == 200
    assert client.get("/season/1900-01").status_code == 404

    page = client.get(f"/season/{SEASON}/gameweek/1").data.decode()
    assert "verdict" in page, "a completed gameweek should show correct/incorrect marks"
    assert "Predicted score" in page


def test_api_returns_json(predicted_db):
    client = create_app(str(predicted_db)).test_client()
    payload = client.get(f"/api/season/{SEASON}/gameweek/1").get_json()
    assert payload["season"] == SEASON
    assert len(payload["matches"]) > 0
    assert payload["matches"][0]["predicted_outcome"] in ("H", "D", "A")


def test_scorelines_are_never_marked_correct(predicted_db):
    """Exact scores are shown side by side and deliberately not scored."""
    with db.connect(predicted_db) as conn:
        data = queries.gameweek(conn, SEASON, 1)
    for match in data["matches"]:
        assert "scoreline_correct" not in match
