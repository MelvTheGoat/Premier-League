"""The rolling retrain loop.

One function does the whole job for one gameweek: take everything known
before that gameweek kicked off, train both models on it, predict the
gameweek, and store the result. Running it once per gameweek all season
is the entire operating procedure — gameweek N's results become
gameweek N+1's training data automatically, because the training set is
defined by a date cutoff rather than by a fixed split.

Nothing is overwritten. Each run records what it was trained on and
writes its own batch of predictions, so a prediction made before kick-off
stays on the record exactly as it was made even after the model has moved
on. That is what makes the history page honest.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from plpredict import config, db
from plpredict.data import ingest
from plpredict.features.build import (
    FEATURE_VERSION,
    build_features,
    feature_columns,
    store_features,
)
from plpredict.models.outcome import CLASSES, OutcomeModel, usable_columns
from plpredict.models.scoreline import ScorelineModel


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class GameweekResult:
    run_id: str
    season: str
    matchday: int
    cutoff: dt.date
    n_training_matches: int
    n_predictions: int
    dropped_columns: int
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (
            f"{self.season} GW{self.matchday}: trained on {self.n_training_matches} "
            f"matches up to {self.cutoff}, predicted {self.n_predictions} fixtures "
            f"(run {self.run_id[:8]})"
        )


def load_match_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Raw results for the goal model, across the competitions it fits."""
    placeholders = ", ".join("?" for _ in config.SCORELINE_MODEL.include_competitions)
    return pd.read_sql(
        f"""
        SELECT season, competition, matchday, match_date,
               home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE competition IN ({placeholders}) AND status = 'played'
        """,
        conn,
        params=list(config.SCORELINE_MODEL.include_competitions),
    )


def gameweek_cutoff(features: pd.DataFrame, season: str, matchday: int) -> dt.date:
    """The first kick-off of a gameweek: the moment predictions must be in by."""
    target = features[(features["season"] == season) & (features["matchday"] == matchday)]
    if target.empty:
        raise ValueError(f"No fixtures found for {season} GW{matchday}")
    return min(dt.date.fromisoformat(value) for value in target["match_date"])


def next_unplayed_gameweek(conn: sqlite3.Connection, season: str) -> int | None:
    """The lowest gameweek in a season that still has a fixture to play."""
    row = conn.execute(
        """
        SELECT MIN(matchday) FROM matches
        WHERE season = ? AND competition = ? AND status != 'played'
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def last_completed_gameweek(conn: sqlite3.Connection, season: str) -> int | None:
    row = conn.execute(
        """
        SELECT MAX(matchday) FROM matches
        WHERE season = ? AND competition = ? AND status = 'played'
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def data_fingerprint(conn: sqlite3.Connection, season: str) -> tuple[int, int | None]:
    """What the pipeline would have to work with, reduced to two numbers.

    A scheduled job runs on a calendar, but fixtures do not: a gameweek
    can be a fortnight away during an international break. Comparing how
    many matches have been played and which gameweek is next tells the
    job whether anything has actually happened since it last published,
    so it can stop before retraining and committing a new database that
    would say exactly the same thing.
    """
    played = conn.execute(
        """
        SELECT COUNT(*) FROM matches
        WHERE season = ? AND competition = ? AND status = 'played'
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()[0]
    return int(played), next_unplayed_gameweek(conn, season)


def published_fingerprint(season: str) -> tuple[int, int | None] | None:
    """The same two numbers for the database the site is currently serving."""
    web_db = Path(config.WEB_DB_PATH)
    if not web_db.is_file():
        return None
    with db.connect(web_db, read_only=True) as conn:
        return data_fingerprint(conn, season)


def train_and_predict(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
    matches: pd.DataFrame,
    season: str,
    matchday: int,
    *,
    save_artifacts: bool = False,
    notes: str | None = None,
    min_feature_coverage: int = 50,
) -> GameweekResult:
    """Train on everything before this gameweek, then predict it."""
    cutoff = gameweek_cutoff(features, season, matchday)

    league = features[features["season"] <= season]
    train = league[
        (pd.to_datetime(league["match_date"]).dt.date < cutoff) & league["result"].notna()
    ]
    target = features[
        (features["season"] == season) & (features["matchday"] == matchday)
    ].copy()
    if train.empty:
        raise ValueError(f"No training data available before {cutoff}")

    candidates = feature_columns(features)
    columns = usable_columns(train, target, candidates, min_feature_coverage)

    outcome = OutcomeModel(config.OUTCOME_MODEL).fit(train, columns)
    probabilities = outcome.predict_frame(target)

    played_before = matches[pd.to_datetime(matches["match_date"]).dt.date < cutoff]
    scoreline = ScorelineModel(config.SCORELINE_MODEL).fit(played_before, cutoff)
    scores = scoreline.predict_frame(
        target, given_outcomes=probabilities["predicted_outcome"]
    )

    run_id = uuid.uuid4().hex
    created_at = _now()

    if save_artifacts:
        directory = Path(config.MODEL_DIR) / f"{season}-GW{matchday:02d}"
        outcome.save(directory / "outcome.pkl")
        scoreline.save(directory / "scoreline.pkl")

    conn.execute(
        """
        INSERT INTO model_runs (
            run_id, created_at, target_season, target_matchday, trained_through,
            n_training_matches, feature_version, outcome_model, scoreline_model,
            metrics, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at,
            season,
            matchday,
            cutoff.isoformat(),
            len(train),
            FEATURE_VERSION,
            "lightgbm+multinomial blend",
            "dixon-coles",
            db.dumps(
                {
                    "n_features": len(columns),
                    "n_candidate_features": len(candidates),
                    "boosting_rounds": outcome.chosen_estimators,
                    "dropped_features": sorted(set(candidates) - set(columns))[:40],
                    "scoreline_home_advantage": scoreline.home_advantage,
                    "scoreline_rho": scoreline.rho,
                    "scoreline_matches": scoreline.n_training_matches,
                    "top_features": outcome.feature_importance(15),
                }
            ),
            notes,
        ),
    )

    rows: list[dict[str, Any]] = []
    for index, record in target.iterrows():
        probability = probabilities.loc[index]
        score = scores.loc[index]
        rows.append(
            {
                "run_id": run_id,
                "match_id": record["match_id"],
                "season": season,
                "matchday": matchday,
                "p_home": float(probability["p_home"]),
                "p_draw": float(probability["p_draw"]),
                "p_away": float(probability["p_away"]),
                "predicted_outcome": str(probability["predicted_outcome"]),
                "exp_home_goals": float(score["exp_home_goals"]),
                "exp_away_goals": float(score["exp_away_goals"]),
                "pred_home_goals": int(score["pred_home_goals"]),
                "pred_away_goals": int(score["pred_away_goals"]),
                "scoreline_prob": float(score["scoreline_prob"]),
                "created_at": created_at,
            }
        )
    db.upsert_many(conn, "predictions", rows, ("run_id", "match_id"))
    db.upsert_many(
        conn,
        "current_predictions",
        [
            {
                "match_id": row["match_id"],
                "run_id": run_id,
                "season": season,
                "matchday": matchday,
            }
            for row in rows
        ],
        ("match_id",),
    )

    result_frame = target[["match_id", "home_team", "away_team", "match_date"]].join(
        probabilities
    ).join(scores)
    return GameweekResult(
        run_id=run_id,
        season=season,
        matchday=matchday,
        cutoff=cutoff,
        n_training_matches=len(train),
        n_predictions=len(rows),
        dropped_columns=len(candidates) - len(columns),
        predictions=result_frame,
    )


def run(
    *,
    season: str | None = None,
    matchday: int | None = None,
    refresh_source: bool = True,
    backfill: bool = False,
    skip_if_unchanged: bool = False,
    db_path: Path | None = None,
    verbose: bool = True,
) -> list[GameweekResult]:
    """Ingest, rebuild features, retrain, predict.

    With ``backfill`` the season's completed gameweeks are replayed in
    order, each predicted by a model that saw only what came before it —
    which is how the history page gets populated on a first run without
    ever showing a prediction made with hindsight.
    """
    report = ingest.run(refresh_source=refresh_source, db_path=db_path)
    if verbose:
        print(f"Ingest: {report.summary()}")

    season = season or config.CURRENT_SEASON

    if skip_if_unchanged:
        published = published_fingerprint(season)
        with db.connect(db_path) as conn:
            current = data_fingerprint(conn, season)
        if published is not None and published == current:
            if verbose:
                print(
                    f"No new results: {current[0]} matches played, "
                    f"gameweek {current[1]} still to come. Nothing to publish."
                )
            return []

    features = build_features(db_path)
    store_features(features, db_path)
    if verbose:
        print(f"Features: {len(features)} matches, {len(feature_columns(features))} columns")

    results: list[GameweekResult] = []

    with db.connect(db_path) as conn:
        matches = load_match_frame(conn)

        # Each target is tagged with whether it is history being filled
        # in or the gameweek about to be played. Only the former is
        # skipped when a prediction already exists: the upcoming
        # gameweek is always re-predicted, because the whole point of a
        # weekly run is that it now has last weekend's results.
        targets: list[tuple[int, bool]] = []
        if matchday is not None:
            targets.append((matchday, False))
        else:
            if backfill:
                completed = last_completed_gameweek(conn, season) or 0
                targets.extend((gameweek, True) for gameweek in range(1, completed + 1))
            upcoming = next_unplayed_gameweek(conn, season)
            if upcoming is not None:
                targets.append((upcoming, False))

        already_predicted = {
            row[1]
            for row in conn.execute(
                "SELECT season, matchday FROM current_predictions WHERE season = ?",
                (season,),
            )
        }

        for target, is_history in targets:
            if is_history and target in already_predicted:
                continue
            result = train_and_predict(conn, features, matches, season, target)
            results.append(result)
            if verbose:
                print(f"  {result.summary()}")

    return results
