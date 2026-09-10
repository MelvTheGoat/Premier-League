"""Fetch → clean → store.

This layer knows about sources and the database and nothing else. It
does no feature work, so a new source can be added (or openfootball
swapped for a paid API) without touching the model.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from plpredict import config, db
from plpredict.data.sources import openfootball
from plpredict.data.sources.openfootball import RawMatch
from plpredict.data.teams import canonical_name


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def make_match_id(
    season: str, competition: str, home: str, away: str, matchday: int | None, stage: str | None
) -> str:
    """A stable id so re-running ingest updates rows instead of duplicating them.

    Built from the fixture's identity rather than a row number, because
    openfootball reorders lines when a match is rescheduled.
    """
    parts = [season, competition, str(matchday or stage or "?"), home, away]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{season}-{competition[:3]}-{digest}"


def outcome_of(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _to_row(match: RawMatch) -> dict:
    home = canonical_name(match.home_team)
    away = canonical_name(match.away_team)
    played = match.is_played
    return {
        "match_id": make_match_id(
            match.season, match.competition, home, away, match.matchday, match.stage
        ),
        "season": match.season,
        "competition": match.competition,
        "matchday": match.matchday,
        "stage": match.stage,
        "match_date": match.date.isoformat() if match.date else None,
        "kickoff": match.kickoff,
        "home_team": home,
        "away_team": away,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "ht_home_goals": match.ht_home_goals,
        "ht_away_goals": match.ht_away_goals,
        "result": outcome_of(match.home_goals, match.away_goals),
        "status": "played" if played else "scheduled",
        "source": "openfootball",
        "updated_at": _now(),
    }


@dataclass
class IngestReport:
    seasons: list[str]
    matches_written: int
    played: int
    scheduled: int
    managers: int
    unavailability: int
    squad_ratings: int
    european: int
    match_stats: int = 0

    def summary(self) -> str:
        return (
            f"{self.matches_written} matches across {len(self.seasons)} seasons "
            f"({self.played} played, {self.scheduled} scheduled); "
            f"{self.managers} manager spells, {self.unavailability} availability rows, "
            f"{self.squad_ratings} squad ratings, {self.european} European entries, "
            f"{self.match_stats} enriched match-stat rows"
        )


def _seasons_to_load(checkout: Path, first_season: str, last_season: str) -> list[str]:
    return [
        season
        for season in openfootball.available_seasons(checkout)
        if first_season <= season <= last_season
    ]


def ingest_matches(
    conn: sqlite3.Connection,
    checkout: Path,
    seasons: list[str],
) -> tuple[int, int, int]:
    rows: list[dict] = []
    for season in seasons:
        for match in openfootball.read_season(checkout, season, config.COMPETITION_FILES):
            rows.append(_to_row(match))

    # A rescheduled fixture can briefly appear twice; keep the last one,
    # which is the version openfootball settled on.
    deduped: dict[str, dict] = {}
    for row in rows:
        deduped[row["match_id"]] = row
    rows = list(deduped.values())

    db.upsert_many(conn, "matches", rows, ("match_id",))
    played = sum(1 for r in rows if r["status"] == "played")
    return len(rows), played, len(rows) - played


# --- manually maintained context ---------------------------------------


def _read_csv(path: Path) -> list[dict]:
    """Read a hand-maintained CSV, ignoring ``#`` comment lines.

    The manual files carry their own documentation at the top, which is
    where a maintainer will actually read it.
    """
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    return [
        {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
        for row in csv.DictReader(lines)
    ]


def ingest_managers(conn: sqlite3.Connection) -> int:
    rows = []
    for row in _read_csv(config.MANUAL_DIR / "managers.csv"):
        team = canonical_name(row.get("team", ""))
        if not team or not row.get("start_date"):
            continue
        rows.append(
            {
                "team": team,
                "manager": row.get("manager") or "unknown",
                "start_date": row["start_date"],
                "end_date": row.get("end_date") or None,
                "source": row.get("source") or "manual",
            }
        )
    return db.upsert_many(conn, "managers", rows, ("team", "start_date"))


def ingest_unavailability(conn: sqlite3.Connection) -> int:
    rows = []
    for row in _read_csv(config.MANUAL_DIR / "unavailability.csv"):
        team = canonical_name(row.get("team", ""))
        if not team or not row.get("season") or not row.get("matchday"):
            continue
        rows.append(
            {
                "season": row["season"],
                "matchday": int(row["matchday"]),
                "team": team,
                "players_out": int(row.get("players_out") or 0),
                "key_players_out": int(row.get("key_players_out") or 0),
                "notes": row.get("notes") or None,
                "source": row.get("source") or "manual",
            }
        )
    return db.upsert_many(conn, "unavailability", rows, ("season", "matchday", "team"))


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def ingest_squad_ratings(conn: sqlite3.Connection) -> int:
    """Squad-quality proxy, one row per club per season.

    Values come from ``data/external/squad_ratings.csv``, which is
    produced by ``scripts/build_squad_ratings.py`` from historical
    FIFA/FC player ratings and can be topped up by hand for seasons the
    ratings dump does not cover.
    """
    rows = []
    for row in _read_csv(config.EXTERNAL_DIR / "squad_ratings.csv"):
        team = canonical_name(row.get("team", ""))
        if not team or not row.get("season"):
            continue
        rows.append(
            {
                "season": row["season"],
                "team": team,
                "overall": _to_float(row.get("overall")),
                "attack": _to_float(row.get("attack")),
                "midfield": _to_float(row.get("midfield")),
                "defence": _to_float(row.get("defence")),
                "top11_mean": _to_float(row.get("top11_mean")),
                "squad_value": _to_float(row.get("squad_value")),
                "source": row.get("source") or "fifa",
            }
        )
    return db.upsert_many(conn, "squad_ratings", rows, ("season", "team"))


def ingest_european_participation(conn: sqlite3.Connection) -> int:
    """Which clubs carry a midweek European commitment in a given season."""
    rows = []
    for row in _read_csv(config.MANUAL_DIR / "european_participation.csv"):
        team = canonical_name(row.get("team", ""))
        if not team or not row.get("season"):
            continue
        rows.append(
            {
                "season": row["season"],
                "team": team,
                "competition": row.get("competition") or "unknown",
            }
        )
    return db.upsert_many(conn, "european_participation", rows, ("season", "team"))


def run(
    *,
    refresh_source: bool = True,
    first_season: str | None = None,
    last_season: str | None = None,
    db_path: Path | None = None,
) -> IngestReport:
    """Full ingest: sync the archive, then load every table."""
    config.ensure_dirs()
    checkout = config.OPENFOOTBALL_CHECKOUT
    if refresh_source:
        openfootball.sync_checkout(checkout, config.OPENFOOTBALL_REPO)
    if not Path(checkout).is_dir():
        raise FileNotFoundError(
            f"No openfootball checkout at {checkout}. "
            "Run with refresh_source=True (the default) to clone it."
        )

    seasons = _seasons_to_load(
        checkout,
        first_season or config.FIRST_TRAINING_SEASON,
        last_season or config.CURRENT_SEASON,
    )
    # The statistics mirror is an enrichment, not a dependency: if it
    # cannot be fetched the pipeline carries on and the feature layer
    # simply drops the columns it would have fed.
    stats_checkout = config.FOOTBALLDATA_CHECKOUT
    if refresh_source:
        try:
            from plpredict.data.sources import footballdata

            footballdata.sync_checkout(stats_checkout, config.FOOTBALLDATA_REPO)
        except Exception as error:  # pragma: no cover - optional source
            print(f"warning: match statistics unavailable ({error})")

    with db.connect(db_path) as conn:
        written, played, scheduled = ingest_matches(conn, checkout, seasons)
        managers = ingest_managers(conn)
        unavailable = ingest_unavailability(conn)
        ratings = ingest_squad_ratings(conn)
        european = ingest_european_participation(conn)
        stats = ingest_match_stats(
            conn, stats_checkout, first_season or config.FIRST_TRAINING_SEASON
        )

    return IngestReport(
        seasons=seasons,
        matches_written=written,
        played=played,
        scheduled=scheduled,
        managers=managers,
        unavailability=unavailable,
        squad_ratings=ratings,
        european=european,
        match_stats=stats,
    )


def ingest_match_stats(
    conn: sqlite3.Connection,
    checkout: Path,
    first_season: str,
) -> int:
    """Attach football-data.co.uk match statistics to known fixtures.

    Joined on season plus the two clubs rather than on the date: the two
    archives occasionally disagree by a day for late kick-offs, but a
    given pairing occurs exactly once per season in a league season.
    """
    from plpredict.data.sources import footballdata

    if not Path(checkout).is_dir():
        return 0

    lookup = {
        (row["season"], row["home_team"], row["away_team"]): row["match_id"]
        for row in conn.execute(
            "SELECT season, home_team, away_team, match_id FROM matches "
            "WHERE competition = ?",
            (config.TARGET_COMPETITION,),
        )
    }

    rows: list[dict] = []
    for record in footballdata.read_all(checkout, first_season=first_season):
        match_id = lookup.get((record.season, record.home_team, record.away_team))
        if match_id is None or not record.stats:
            continue
        row = {"match_id": match_id, "season": record.season}
        row.update({name: record.stats.get(name) for name in footballdata.STAT_COLUMNS.values()})
        row["referee"] = record.referee
        row["source"] = "football-data.co.uk"
        rows.append(row)

    return db.upsert_many(conn, "match_stats", rows, ("match_id",))
