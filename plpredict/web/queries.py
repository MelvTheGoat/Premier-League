"""Read-only views over the database, shaped for the pages that use them.

The web layer never trains or predicts. It reads what the pipeline last
stored, which is what lets the scheduled job and the site run
independently: the job can be mid-retrain and the site keeps serving the
previous gameweek's published predictions.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any

from plpredict import config, db
from plpredict.data.teams import display_name

_OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _row_to_match(row: sqlite3.Row) -> dict[str, Any]:
    played = row["home_goals"] is not None and row["away_goals"] is not None
    predicted = row["predicted_outcome"]
    probabilities = (
        {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
        if predicted is not None
        else {}
    )
    correct = None
    if played and predicted is not None:
        correct = predicted == row["result"]

    return {
        "match_id": row["match_id"],
        "date": row["match_date"],
        "kickoff": row["kickoff"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_short": display_name(row["home_team"]),
        "away_short": display_name(row["away_team"]),
        "played": played,
        "home_goals": row["home_goals"],
        "away_goals": row["away_goals"],
        "result": row["result"],
        "result_label": _OUTCOME_LABELS.get(row["result"] or "", None),
        "has_prediction": predicted is not None,
        "predicted_outcome": predicted,
        "predicted_label": _OUTCOME_LABELS.get(predicted or "", None),
        "probabilities": probabilities,
        "confidence": max(probabilities.values()) if probabilities else None,
        "pred_home_goals": row["pred_home_goals"],
        "pred_away_goals": row["pred_away_goals"],
        "exp_home_goals": row["exp_home_goals"],
        "exp_away_goals": row["exp_away_goals"],
        "scoreline_prob": row["scoreline_prob"],
        "outcome_correct": correct,
        "run_id": row["run_id"],
    }


_GAMEWEEK_QUERY = """
    SELECT m.match_id, m.match_date, m.kickoff, m.home_team, m.away_team,
           m.home_goals, m.away_goals, m.result,
           p.predicted_outcome, p.p_home, p.p_draw, p.p_away,
           p.pred_home_goals, p.pred_away_goals,
           p.exp_home_goals, p.exp_away_goals, p.scoreline_prob, p.run_id
    FROM matches m
    LEFT JOIN current_predictions cp ON cp.match_id = m.match_id
    LEFT JOIN predictions p
           ON p.run_id = cp.run_id AND p.match_id = cp.match_id
    WHERE m.season = ? AND m.competition = ? AND m.matchday = ?
    ORDER BY m.match_date, m.kickoff, m.home_team
"""


def gameweek(conn: sqlite3.Connection, season: str, matchday: int) -> dict[str, Any]:
    rows = conn.execute(
        _GAMEWEEK_QUERY, (season, config.TARGET_COMPETITION, matchday)
    ).fetchall()
    matches = [_row_to_match(row) for row in rows]
    scored = [match for match in matches if match["outcome_correct"] is not None]
    exact = [
        match
        for match in scored
        if match["pred_home_goals"] is not None
        and match["pred_home_goals"] == match["home_goals"]
        and match["pred_away_goals"] == match["away_goals"]
    ]
    return {
        "season": season,
        "matchday": matchday,
        "matches": matches,
        "is_complete": bool(matches) and all(match["played"] for match in matches),
        "any_played": any(match["played"] for match in matches),
        "n_scored": len(scored),
        "n_correct": sum(1 for match in scored if match["outcome_correct"]),
        "n_exact_scores": len(exact),
        "accuracy": (
            sum(1 for match in scored if match["outcome_correct"]) / len(scored)
            if scored
            else None
        ),
        "dates": sorted({match["date"] for match in matches if match["date"]}),
    }


def season_gameweeks(conn: sqlite3.Connection, season: str) -> list[dict[str, Any]]:
    """One summary row per gameweek, for the season navigator."""
    rows = conn.execute(
        """
        SELECT m.matchday,
               COUNT(*) AS n,
               SUM(m.status = 'played') AS played,
               MIN(m.match_date) AS first_date,
               MAX(m.match_date) AS last_date,
               SUM(CASE WHEN p.predicted_outcome IS NOT NULL THEN 1 ELSE 0 END) AS predicted,
               SUM(CASE WHEN m.result IS NOT NULL
                         AND p.predicted_outcome = m.result THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN m.result IS NOT NULL
                         AND p.predicted_outcome IS NOT NULL THEN 1 ELSE 0 END) AS scored
        FROM matches m
        LEFT JOIN current_predictions cp ON cp.match_id = m.match_id
        LEFT JOIN predictions p
               ON p.run_id = cp.run_id AND p.match_id = cp.match_id
        WHERE m.season = ? AND m.competition = ?
        GROUP BY m.matchday
        ORDER BY m.matchday
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchall()
    return [
        {
            "matchday": row["matchday"],
            "n": row["n"],
            "played": row["played"] or 0,
            "is_complete": (row["played"] or 0) == row["n"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "has_predictions": (row["predicted"] or 0) > 0,
            "n_scored": row["scored"] or 0,
            "n_correct": row["correct"] or 0,
            "accuracy": (row["correct"] / row["scored"]) if row["scored"] else None,
        }
        for row in rows
    ]


def current_matchday(conn: sqlite3.Connection, season: str) -> int:
    """The gameweek the site should open on: the next one still to be played."""
    row = conn.execute(
        """
        SELECT MIN(matchday) FROM matches
        WHERE season = ? AND competition = ? AND status != 'played'
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    row = conn.execute(
        """
        SELECT MAX(matchday) FROM matches WHERE season = ? AND competition = ?
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 1


def available_seasons(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT season FROM matches WHERE competition = ? ORDER BY season DESC",
            (config.TARGET_COMPETITION,),
        )
    ]


def season_record(conn: sqlite3.Connection, season: str) -> dict[str, Any]:
    """Running accuracy for the season so far."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS scored,
               SUM(CASE WHEN p.predicted_outcome = m.result THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN p.pred_home_goals = m.home_goals
                         AND p.pred_away_goals = m.away_goals THEN 1 ELSE 0 END) AS exact
        FROM matches m
        JOIN current_predictions cp ON cp.match_id = m.match_id
        JOIN predictions p ON p.run_id = cp.run_id AND p.match_id = cp.match_id
        WHERE m.season = ? AND m.competition = ? AND m.result IS NOT NULL
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    scored = row["scored"] or 0
    return {
        "scored": scored,
        "correct": row["correct"] or 0,
        "exact": row["exact"] or 0,
        "accuracy": (row["correct"] / scored) if scored else None,
        "exact_rate": (row["exact"] / scored) if scored else None,
    }


def latest_run(conn: sqlite3.Connection, season: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM model_runs WHERE target_season = ?
        ORDER BY created_at DESC, target_matchday DESC LIMIT 1
        """,
        (season,),
    ).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["metrics"] = db.loads(row["metrics"]) or {}
    return record


def data_freshness(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT MAX(updated_at) AS updated, COUNT(*) AS n FROM matches"
    ).fetchone()
    latest_result = conn.execute(
        """
        SELECT MAX(match_date) FROM matches
        WHERE competition = ? AND status = 'played'
        """,
        (config.TARGET_COMPETITION,),
    ).fetchone()
    return {
        "ingested_at": row["updated"],
        "n_matches": row["n"],
        "latest_result_date": latest_result[0] if latest_result else None,
        "today": dt.date.today().isoformat(),
    }
