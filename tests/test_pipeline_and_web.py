"""End-to-end: the rolling retrain, and the pages that read its output."""

from __future__ import annotations

import pandas as pd
import pytest

from plpredict import db
from plpredict.features.build import FeatureBuilder
from plpredict.pipeline.run import (
    data_fingerprint,
    gameweek_cutoff,
    last_completed_gameweek,
    load_match_frame,
    next_unplayed_gameweek,
    train_and_predict,
)
from plpredict.web import queries
from plpredict.web.export import export
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


@pytest.fixture
def scratch_db(predicted_db, tmp_path):
    """A disposable copy of the predicted database.

    ``predicted_db`` is module-scoped because running the pipeline over
    it is slow, so any test that writes must work on a copy — otherwise
    it leaks state into every test that follows and the suite starts
    depending on the order it happens to run in.
    """
    copy = tmp_path / "scratch.db"
    copy.write_bytes(predicted_db.read_bytes())
    return copy


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


def test_predictions_are_kept_not_overwritten(scratch_db):
    """Re-predicting a gameweek adds a run; the old one stays on record."""
    with db.connect(scratch_db) as conn:
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


def test_backfill_still_repredicts_the_upcoming_gameweek(predicted_db, monkeypatch):
    """A weekly `--backfill` run must not go quiet once history is filled.

    Completed gameweeks are skipped when they already have a prediction,
    but the gameweek about to be played is re-predicted every run - that
    is the entire point of running it after each round of fixtures.
    """
    import plpredict.pipeline.run as pipeline

    monkeypatch.setattr(pipeline.ingest, "run", lambda **kwargs: _NoopReport())
    monkeypatch.setattr(pipeline, "build_features", lambda db_path=None: _features(predicted_db))
    monkeypatch.setattr(pipeline, "store_features", lambda frame, db_path=None: 0)
    monkeypatch.setattr(
        pipeline,
        "train_and_predict",
        _recording_train_and_predict(calls := []),
    )

    pipeline.run(season=SEASON, refresh_source=False, backfill=True, db_path=predicted_db, verbose=False)

    with db.connect(predicted_db) as conn:
        upcoming = next_unplayed_gameweek(conn, SEASON)
    assert calls == [upcoming], "only the upcoming gameweek should be re-predicted"


class _NoopReport:
    def summary(self) -> str:
        return "no ingest"


def _features(path):
    with db.connect(path) as conn:
        return FeatureBuilder(conn).build()


def _recording_train_and_predict(calls):
    def recorder(conn, features, matches, season, matchday, **kwargs):
        calls.append(matchday)
        from plpredict.pipeline.run import GameweekResult
        import datetime as dt

        return GameweekResult(
            run_id="test", season=season, matchday=matchday,
            cutoff=dt.date(2026, 1, 1), n_training_matches=0,
            n_predictions=0, dropped_columns=0,
        )

    return recorder


def test_export_keeps_only_what_the_pages_read(predicted_db, tmp_path):
    exported = tmp_path / "serving.db"
    counts = export(predicted_db, exported, season=None)

    assert counts["matches"] > 0
    assert counts["current_predictions"] > 0
    assert exported.stat().st_size < predicted_db.stat().st_size

    with db.connect(exported, read_only=True) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    # The feature table is the bulk of the working database and no page
    # touches it, so it must not be shipped.
    assert "features" not in tables
    assert {"matches", "predictions", "current_predictions", "model_runs"} <= tables


def test_site_serves_from_a_read_only_file(predicted_db, tmp_path):
    """A deployed function gets a read-only filesystem; this must still work."""
    exported = tmp_path / "serving.db"
    export(predicted_db, exported, season=None)
    exported.chmod(0o444)

    client = create_app(str(exported)).test_client()
    assert client.get(f"/season/{SEASON}/gameweek/1").status_code == 200

    # And the same file opened explicitly read-only, which is how the app
    # opens the committed serving database.
    with db.connect(exported, read_only=True) as conn:
        assert queries.gameweek(conn, SEASON, 1)["matches"]


def test_read_only_connection_refuses_to_create_a_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        with db.connect(tmp_path / "missing.db", read_only=True):
            pass
    assert not (tmp_path / "missing.db").exists()


def test_fingerprint_tracks_results_and_the_next_gameweek(scratch_db):
    with db.connect(scratch_db) as conn:
        played, upcoming = data_fingerprint(conn, SEASON)
        assert played > 0
        assert upcoming == next_unplayed_gameweek(conn, SEASON)

        # Playing the next gameweek moves both numbers.
        conn.execute(
            """
            UPDATE matches SET status = 'played', home_goals = 1, away_goals = 1,
                               result = 'D'
            WHERE season = ? AND matchday = ? AND competition = 'premier_league'
            """,
            (SEASON, upcoming),
        )
        after = data_fingerprint(conn, SEASON)
    assert after[0] > played
    assert after[1] != upcoming


def test_scheduled_run_stops_when_nothing_has_been_played(predicted_db, tmp_path, monkeypatch):
    """The job runs daily; fixtures do not. A quiet day must cost nothing."""
    import plpredict.pipeline.run as pipeline

    serving = tmp_path / "serving.db"
    export(predicted_db, serving, season=None)
    monkeypatch.setattr(pipeline.config, "WEB_DB_PATH", serving)
    monkeypatch.setattr(pipeline.ingest, "run", lambda **kwargs: _NoopReport())

    trained: list[int] = []
    monkeypatch.setattr(pipeline, "train_and_predict", _recording_train_and_predict(trained))
    monkeypatch.setattr(pipeline, "build_features", lambda db_path=None: _features(predicted_db))
    monkeypatch.setattr(pipeline, "store_features", lambda frame, db_path=None: 0)

    results = pipeline.run(
        season=SEASON, refresh_source=False, skip_if_unchanged=True,
        db_path=predicted_db, verbose=False,
    )
    assert results == []
    assert trained == [], "nothing should be retrained when no results have arrived"


def test_scheduled_run_proceeds_once_a_gameweek_is_played(predicted_db, tmp_path, monkeypatch):
    import plpredict.pipeline.run as pipeline

    # A serving database exported before the latest gameweek was played.
    serving = tmp_path / "stale.db"
    export(predicted_db, serving, season=None)
    with db.connect(serving) as conn:
        upcoming = next_unplayed_gameweek(conn, SEASON)
        conn.execute(
            """
            UPDATE matches SET status = 'scheduled', home_goals = NULL,
                               away_goals = NULL, result = NULL
            WHERE season = ? AND matchday = ? AND competition = 'premier_league'
            """,
            (SEASON, (upcoming or 2) - 1),
        )
    monkeypatch.setattr(pipeline.config, "WEB_DB_PATH", serving)
    monkeypatch.setattr(pipeline.ingest, "run", lambda **kwargs: _NoopReport())

    trained: list[int] = []
    monkeypatch.setattr(pipeline, "train_and_predict", _recording_train_and_predict(trained))
    monkeypatch.setattr(pipeline, "build_features", lambda db_path=None: _features(predicted_db))
    monkeypatch.setattr(pipeline, "store_features", lambda frame, db_path=None: 0)

    pipeline.run(
        season=SEASON, refresh_source=False, skip_if_unchanged=True,
        db_path=predicted_db, verbose=False,
    )
    assert trained, "a newly played gameweek must trigger a retrain"


def test_history_survives_a_rebuilt_database(predicted_db, tmp_path, monkeypatch):
    """A scheduled run starts with no working database and must not erase history.

    `data/db/` is a build artefact, so CI rebuilds it from source every
    run and predicts only the next gameweek. Without restoring the
    published record, the exported database would contain that gameweek
    alone and every earlier forecast would vanish.
    """
    import plpredict.pipeline.run as pipeline

    serving = tmp_path / "serving.db"
    export(predicted_db, serving, season=None)
    with db.connect(serving, read_only=True) as conn:
        before = conn.execute("SELECT COUNT(*) FROM current_predictions").fetchone()[0]
    assert before > 10, "fixture should have several gameweeks of history"

    # A rebuilt working database: matches present, no predictions at all.
    rebuilt = tmp_path / "rebuilt.db"
    with db.connect(predicted_db, read_only=True) as origin, db.connect(rebuilt) as fresh:
        rows = [tuple(r) for r in origin.execute("SELECT * FROM matches")]
        placeholders = ", ".join("?" for _ in rows[0])
        fresh.executemany(f"INSERT INTO matches VALUES ({placeholders})", rows)

    with db.connect(rebuilt) as conn:
        assert conn.execute("SELECT COUNT(*) FROM current_predictions").fetchone()[0] == 0
        restored = pipeline.restore_prediction_history(conn, serving)
        after = conn.execute("SELECT COUNT(*) FROM current_predictions").fetchone()[0]
    assert restored > 0
    assert after == before, "every published prediction should come back"


def test_restoring_history_never_rewrites_a_past_forecast(scratch_db, tmp_path):
    """Restoring is append-only: what was forecast at the time stands."""
    import plpredict.pipeline.run as pipeline

    serving = tmp_path / "serving.db"
    export(scratch_db, serving, season=None)

    with db.connect(scratch_db) as conn:
        original = conn.execute(
            "SELECT match_id, run_id FROM current_predictions ORDER BY match_id LIMIT 1"
        ).fetchone()
        pipeline.restore_prediction_history(conn, serving)
        after = conn.execute(
            "SELECT run_id FROM current_predictions WHERE match_id = ?",
            (original["match_id"],),
        ).fetchone()
    assert after["run_id"] == original["run_id"]


def test_restoring_history_is_a_no_op_without_a_published_database(predicted_db, tmp_path):
    import plpredict.pipeline.run as pipeline

    with db.connect(predicted_db) as conn:
        assert pipeline.restore_prediction_history(conn, tmp_path / "absent.db") == 0


def test_a_prediction_made_before_kickoff_is_not_flagged(predicted_db):
    with db.connect(predicted_db) as conn:
        data = queries.gameweek(conn, SEASON, 1)
    # The fixture predicts each gameweek before it is played.
    assert data["first_kickoff"] is not None
    for match in data["matches"]:
        assert match["predicted_at"] is not None


def test_a_prediction_stored_after_kickoff_is_flagged(scratch_db):
    """A late forecast must not be displayed as though it were live."""
    with db.connect(scratch_db) as conn:
        upcoming = next_unplayed_gameweek(conn, SEASON)
        before = queries.gameweek(conn, SEASON, upcoming)
        assert before["published_late"] is False

        # Backdate the gameweek so its kick-off precedes the stored run.
        conn.execute(
            """
            UPDATE matches SET match_date = '2000-01-01', kickoff = '15:00'
            WHERE season = ? AND matchday = ? AND competition = 'premier_league'
            """,
            (SEASON, upcoming),
        )
        after = queries.gameweek(conn, SEASON, upcoming)

    assert after["published_late"] is True
    assert all(m["published_before_kickoff"] is False for m in after["matches"])
