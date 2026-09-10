"""SQLite storage for raw data, features, predictions and model runs.

One file, four concerns, deliberately kept apart:

``matches``        every fixture the pipeline knows about, played or not
``managers`` / ``unavailability`` / ``squad_ratings``
                   the contextual inputs that are not derivable from results
``features``       the versioned, point-in-time feature table
``predictions`` / ``model_runs``
                   what each trained model said, and what it was trained on

Predictions are never overwritten in place: every retrain writes a new
``model_runs`` row and a new batch of predictions, so the history page
can always show what was actually forecast before a gameweek kicked off
rather than what the current model would say with hindsight.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from plpredict import config

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    season          TEXT NOT NULL,
    competition     TEXT NOT NULL,
    matchday        INTEGER,
    stage           TEXT,
    match_date      TEXT,
    kickoff         TEXT,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_goals      INTEGER,
    away_goals      INTEGER,
    ht_home_goals   INTEGER,
    ht_away_goals   INTEGER,
    result          TEXT,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    source          TEXT,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_season_gw
    ON matches (season, competition, matchday);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home ON matches (home_team, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches (away_team, match_date);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id             TEXT PRIMARY KEY REFERENCES matches (match_id) ON DELETE CASCADE,
    season               TEXT NOT NULL,
    home_shots           REAL,
    away_shots           REAL,
    home_shots_on_target REAL,
    away_shots_on_target REAL,
    home_corners         REAL,
    away_corners         REAL,
    home_fouls           REAL,
    away_fouls           REAL,
    home_yellows         REAL,
    away_yellows         REAL,
    home_reds            REAL,
    away_reds            REAL,
    referee              TEXT,
    source               TEXT
);
CREATE INDEX IF NOT EXISTS idx_match_stats_season ON match_stats (season);

CREATE TABLE IF NOT EXISTS managers (
    team        TEXT NOT NULL,
    manager     TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT,
    source      TEXT,
    PRIMARY KEY (team, start_date)
);

CREATE TABLE IF NOT EXISTS unavailability (
    season          TEXT NOT NULL,
    matchday        INTEGER NOT NULL,
    team            TEXT NOT NULL,
    players_out     INTEGER NOT NULL DEFAULT 0,
    key_players_out INTEGER NOT NULL DEFAULT 0,
    notes           TEXT,
    source          TEXT,
    PRIMARY KEY (season, matchday, team)
);

CREATE TABLE IF NOT EXISTS squad_ratings (
    season      TEXT NOT NULL,
    team        TEXT NOT NULL,
    overall     REAL,
    attack      REAL,
    midfield    REAL,
    defence     REAL,
    top11_mean  REAL,
    squad_value REAL,
    source      TEXT,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS european_participation (
    season      TEXT NOT NULL,
    team        TEXT NOT NULL,
    competition TEXT NOT NULL,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS features (
    match_id        TEXT PRIMARY KEY REFERENCES matches (match_id) ON DELETE CASCADE,
    season          TEXT NOT NULL,
    matchday        INTEGER,
    match_date      TEXT,
    feature_version TEXT NOT NULL,
    built_at        TEXT NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_features_season_gw ON features (season, matchday);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    target_season       TEXT NOT NULL,
    target_matchday     INTEGER NOT NULL,
    trained_through     TEXT,
    n_training_matches  INTEGER,
    feature_version     TEXT,
    outcome_model       TEXT,
    scoreline_model     TEXT,
    metrics             TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_target ON model_runs (target_season, target_matchday);

CREATE TABLE IF NOT EXISTS predictions (
    run_id            TEXT NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    match_id          TEXT NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_pred_season_gw ON predictions (season, matchday);

-- The prediction of record for each match: the one made by the newest
-- run that was trained strictly before that gameweek kicked off.
CREATE TABLE IF NOT EXISTS current_predictions (
    match_id   TEXT PRIMARY KEY REFERENCES matches (match_id) ON DELETE CASCADE,
    run_id     TEXT NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    season     TEXT NOT NULL,
    matchday   INTEGER NOT NULL
);
"""


@contextmanager
def connect(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database, creating its directory and schema on demand."""
    db_path = Path(path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_many(
    conn: sqlite3.Connection,
    table: str,
    rows: Iterable[dict[str, Any]],
    key_columns: tuple[str, ...],
) -> int:
    """Insert rows, updating the non-key columns of any that already exist."""
    rows = list(rows)
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{col} = excluded.{col}" for col in columns if col not in key_columns
    )
    conflict = ", ".join(key_columns)
    statement = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        if updates
        else f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(statement, [tuple(row[col] for col in columns) for row in rows])
    return len(rows)


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def loads(value: str | None) -> Any:
    return json.loads(value) if value else None
