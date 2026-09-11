"""Build the small, read-only database the website serves from.

The working database carries everything the pipeline needs: the feature
table, match statistics, every competition's fixtures. The site reads
almost none of it — around 53 MB of a 61 MB file is the feature table,
which no page touches.

This exports just the serving surface: Premier League fixtures, the
predictions of record, and the model-run metadata behind the model page.
The result is small enough to commit and to ship inside a serverless
deployment, and it is safe to open read-only because nothing in it is
ever written at request time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from plpredict import config, db

SERVING_SCHEMA = """
CREATE TABLE matches (
    match_id      TEXT PRIMARY KEY,
    season        TEXT NOT NULL,
    competition   TEXT NOT NULL,
    matchday      INTEGER,
    match_date    TEXT,
    kickoff       TEXT,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    home_goals    INTEGER,
    away_goals    INTEGER,
    result        TEXT,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX idx_matches_season_gw ON matches (season, competition, matchday);

CREATE TABLE model_runs (
    run_id             TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    target_season      TEXT NOT NULL,
    target_matchday    INTEGER NOT NULL,
    trained_through    TEXT,
    n_training_matches INTEGER,
    feature_version    TEXT,
    outcome_model      TEXT,
    scoreline_model    TEXT,
    metrics            TEXT,
    notes              TEXT
);

CREATE TABLE predictions (
    run_id            TEXT NOT NULL,
    match_id          TEXT NOT NULL,
    season            TEXT NOT NULL,
    matchday          INTEGER NOT NULL,
    p_home            REAL NOT NULL,
    p_draw            REAL NOT NULL,
    p_away            REAL NOT NULL,
    predicted_outcome TEXT NOT NULL,
    exp_home_goals    REAL,
    exp_away_goals    REAL,
    pred_home_goals   INTEGER,
    pred_away_goals   INTEGER,
    scoreline_prob    REAL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (run_id, match_id)
);
CREATE INDEX idx_pred_season_gw ON predictions (season, matchday);

CREATE TABLE current_predictions (
    match_id TEXT PRIMARY KEY,
    run_id   TEXT NOT NULL,
    season   TEXT NOT NULL,
    matchday INTEGER NOT NULL
);
"""

MATCH_COLUMNS = (
    "match_id, season, competition, matchday, match_date, kickoff, "
    "home_team, away_team, home_goals, away_goals, result, status, updated_at"
)


def export(source: Path, destination: Path, season: str | None) -> dict[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    target = sqlite3.connect(destination)
    target.executescript(SERVING_SCHEMA)

    counts: dict[str, int] = {}
    with db.connect(source, read_only=True) as origin:
        season_filter = " AND season = ?" if season else ""
        params: tuple = (config.TARGET_COMPETITION,) + ((season,) if season else ())

        rows = origin.execute(
            f"SELECT {MATCH_COLUMNS} FROM matches "
            f"WHERE competition = ?{season_filter}",
            params,
        ).fetchall()
        target.executemany(
            f"INSERT INTO matches ({MATCH_COLUMNS}) VALUES "
            f"({', '.join('?' for _ in MATCH_COLUMNS.split(','))})",
            [tuple(row) for row in rows],
        )
        counts["matches"] = len(rows)

        for table in ("model_runs", "predictions", "current_predictions"):
            column = "target_season" if table == "model_runs" else "season"
            where = f" WHERE {column} = ?" if season else ""
            rows = origin.execute(
                f"SELECT * FROM {table}{where}", (season,) if season else ()
            ).fetchall()
            if rows:
                placeholders = ", ".join("?" for _ in rows[0].keys())
                target.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    [tuple(row) for row in rows],
                )
            counts[table] = len(rows)

    target.commit()
    # Drop the write-ahead log and compact, so the committed file is as
    # small as it can be and is a single self-contained artefact.
    target.execute("PRAGMA journal_mode = DELETE")
    target.execute("VACUUM")
    target.close()
    return counts


