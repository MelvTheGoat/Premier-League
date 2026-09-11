"""Flask app serving the predictions the pipeline has already stored.

Deliberately thin: no model is loaded here and nothing is computed on a
request. Every page is a query against the same tables the pipeline
writes, so the site updates by itself the moment the scheduled job
finishes a gameweek.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, url_for

from plpredict import config, db
from plpredict.web import queries


def resolve_database(db_path: str | None = None) -> tuple[Path, bool]:
    """Pick the database to serve, and whether to open it read-only.

    Preference order is an explicitly supplied path, then the committed
    serving database, then the pipeline's working database. Only the
    serving database is opened read-only: it is exported without a
    write-ahead log precisely so it can be, which is what lets the site
    run on a host with a read-only filesystem. The working database is
    in WAL mode, where an immutable read could miss a commit the
    pipeline has just made.
    """
    if db_path:
        return Path(db_path), False
    web_db = Path(config.WEB_DB_PATH)
    if web_db.is_file():
        return web_db, True
    return Path(config.DB_PATH), False


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    resolved, read_only = resolve_database(db_path)
    app.config["DB_PATH"] = str(resolved)
    app.config["DB_READ_ONLY"] = read_only

    def _connection():
        return db.connect(app.config["DB_PATH"], read_only=app.config["DB_READ_ONLY"])

    @app.template_filter("percent")
    def percent(value: float | None, places: int = 0) -> str:
        return "-" if value is None else f"{value * 100:.{places}f}%"

    @app.template_filter("pretty_date")
    def pretty_date(value: str | None) -> str:
        if not value:
            return ""
        import datetime as dt

        try:
            parsed = dt.date.fromisoformat(value)
        except ValueError:
            return value
        return parsed.strftime("%a %-d %b")

    @app.route("/")
    def index():
        with _connection() as conn:
            season = config.CURRENT_SEASON
            return redirect(
                url_for(
                    "gameweek",
                    season=season,
                    matchday=queries.current_matchday(conn, season),
                )
            )

    @app.route("/season/<season>/gameweek/<int:matchday>")
    def gameweek(season: str, matchday: int):
        with _connection() as conn:
            data = queries.gameweek(conn, season, matchday)
            if not data["matches"]:
                abort(404)
            weeks = queries.season_gameweeks(conn, season)
            return render_template(
                "gameweek.html",
                data=data,
                weeks=weeks,
                season=season,
                seasons=queries.available_seasons(conn),
                record=queries.season_record(conn, season),
                current=queries.current_matchday(conn, season),
                freshness=queries.data_freshness(conn),
            )

    @app.route("/season/<season>")
    def season_overview(season: str):
        with _connection() as conn:
            weeks = queries.season_gameweeks(conn, season)
            if not weeks:
                abort(404)
            return render_template(
                "season.html",
                weeks=weeks,
                season=season,
                seasons=queries.available_seasons(conn),
                record=queries.season_record(conn, season),
                current=queries.current_matchday(conn, season),
                freshness=queries.data_freshness(conn),
            )

    @app.route("/model")
    def model_card():
        with _connection() as conn:
            season = config.CURRENT_SEASON
            return render_template(
                "model.html",
                season=season,
                seasons=queries.available_seasons(conn),
                run=queries.latest_run(conn, season),
                record=queries.season_record(conn, season),
                current=queries.current_matchday(conn, season),
                freshness=queries.data_freshness(conn),
            )

    @app.route("/api/season/<season>/gameweek/<int:matchday>")
    def api_gameweek(season: str, matchday: int) -> Any:
        with _connection() as conn:
            data = queries.gameweek(conn, season, matchday)
            if not data["matches"]:
                abort(404)
            return jsonify(data)

    @app.route("/api/season/<season>")
    def api_season(season: str) -> Any:
        with _connection() as conn:
            return jsonify(
                {
                    "season": season,
                    "record": queries.season_record(conn, season),
                    "gameweeks": queries.season_gameweeks(conn, season),
                }
            )

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True, port=5000)
