"""Assemble the versioned, point-in-time feature table.

The philosophy this project is built on is that a match's *context* —
who is missing, who just changed manager, what each side still has to
play for, how many games they have had in the last fortnight — belongs
in the model as ordinary features that the model weighs for itself,
alongside goals and points. Nothing in here applies a hand-tuned
adjustment to a prediction; every contextual signal is turned into a
number and handed to the learner.

Two structural rules make that safe:

**One chronological pass.** State moves forward only. A feature cannot
see a result that has not been fed in yet, so leakage is prevented by
construction rather than by discipline.

**Gameweek granularity.** Features for every match in gameweek N are
computed from the state as it stood before the *first* kick-off of that
gameweek, never from the Saturday results when the fixture is on the
Monday. That is exactly the information the platform has when it
publishes its predictions, so training and serving see the same thing.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from plpredict import config, db
from plpredict.features.elo import COMPETITION_WEIGHTS, EloTable
from plpredict.features.state import HeadToHead, LeagueTable, MatchOutcome, TeamState

# Bumped whenever the meaning of a column changes, so stale rows are
# never silently mixed with fresh ones.
FEATURE_VERSION = "2026.09.1"

_LEAGUE_COMPETITIONS = {"premier_league", "championship", "league_one"}
_EUROPEAN_TIERS = {
    "champions_league": 3.0,
    "europa_league": 2.0,
    "conference_league": 1.0,
}


@dataclass
class MatchRow:
    match_id: str
    season: str
    competition: str
    matchday: int | None
    match_date: dt.date
    kickoff: str | None
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    status: str
    stats: dict[str, float]


def _as_date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:
        return None


def load_matches(conn: sqlite3.Connection) -> list[MatchRow]:
    """Every fixture the pipeline knows about, with stats joined on."""
    query = """
        SELECT m.*, s.home_shots, s.away_shots,
               s.home_shots_on_target, s.away_shots_on_target,
               s.home_corners, s.away_corners
        FROM matches m
        LEFT JOIN match_stats s ON s.match_id = m.match_id
        ORDER BY m.match_date, m.kickoff
    """
    rows: list[MatchRow] = []
    for record in conn.execute(query):
        date = _as_date(record["match_date"])
        if date is None:
            continue
        stats = {
            name: record[name]
            for name in (
                "home_shots",
                "away_shots",
                "home_shots_on_target",
                "away_shots_on_target",
                "home_corners",
                "away_corners",
            )
            if record[name] is not None
        }
        rows.append(
            MatchRow(
                match_id=record["match_id"],
                season=record["season"],
                competition=record["competition"],
                matchday=record["matchday"],
                match_date=date,
                kickoff=record["kickoff"],
                home_team=record["home_team"],
                away_team=record["away_team"],
                home_goals=record["home_goals"],
                away_goals=record["away_goals"],
                status=record["status"],
                stats=stats,
            )
        )
    return rows


# --- contextual lookups -------------------------------------------------


def _load_managers(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    spells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conn.execute("SELECT * FROM managers ORDER BY team, start_date"):
        start = _as_date(row["start_date"])
        if start is None:
            continue
        spells[row["team"]].append(
            {"manager": row["manager"], "start": start, "end": _as_date(row["end_date"])}
        )
    return spells


def _load_squad_ratings(conn: sqlite3.Connection) -> dict[tuple[str, str], dict[str, Any]]:
    """Squad ratings, carried forward with a record of how stale they are.

    EA rate squads once a season and the current season's edition is not
    published in a usable form until well after kick-off, so the freshest
    rating available for a live gameweek is usually a season old. Rather
    than dropping the signal, the most recent rating is carried forward
    and its age is handed to the model as its own feature, letting the
    model discount a stale rating on its own terms.
    """
    by_team: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seasons: set[str] = set()
    for row in conn.execute("SELECT * FROM squad_ratings"):
        by_team[row["team"]][row["season"]] = dict(row)
        seasons.add(row["season"])
    for row in conn.execute("SELECT DISTINCT season FROM matches"):
        seasons.add(row["season"])

    ordered = sorted(seasons)
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for team, ratings in by_team.items():
        latest: dict[str, Any] | None = None
        latest_season: str | None = None
        for season in ordered:
            if season in ratings:
                latest, latest_season = ratings[season], season
            if latest is not None:
                age = ordered.index(season) - ordered.index(latest_season)
                resolved[(season, team)] = {**latest, "rating_age_seasons": float(age)}
    return resolved


def _season_z_scores(
    ratings: dict[tuple[str, str], dict[str, Any]], key: str
) -> dict[str, tuple[float, float]]:
    """Mean and spread of a rating within each season, for normalisation.

    Ratings inflate over time (the same club scores higher in FC 26 than
    in FIFA 15), so the absolute number is not comparable across seasons.
    Standardising within a season keeps "how good is this squad relative
    to the rest of the league" comparable, which is the thing that
    matters.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for (season, _), rating in ratings.items():
        value = rating.get(key)
        if value is not None:
            grouped[season].append(float(value))
    stats: dict[str, tuple[float, float]] = {}
    for season, values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        stats[season] = (mean, variance**0.5 or 1.0)
    return stats


def _load_unavailability(conn: sqlite3.Connection) -> dict[tuple[str, int, str], sqlite3.Row]:
    return {
        (row["season"], row["matchday"], row["team"]): row
        for row in conn.execute("SELECT * FROM unavailability")
    }


def _load_european(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    return {
        (row["season"], row["team"]): row["competition"]
        for row in conn.execute("SELECT * FROM european_participation")
    }


# --- the chronological pass --------------------------------------------


def _gameweek_blocks(matches: list[MatchRow]) -> list[tuple[dt.date, str, Any]]:
    """Order the timeline as gameweek blocks plus individual other matches.

    Premier League matches are grouped by gameweek and the whole group is
    positioned at its earliest kick-off, so every match in a gameweek is
    featurised from the same pre-gameweek state.
    """
    blocks: dict[tuple[str, int], list[MatchRow]] = defaultdict(list)
    others: list[MatchRow] = []
    for match in matches:
        if match.competition == config.TARGET_COMPETITION and match.matchday is not None:
            blocks[(match.season, match.matchday)].append(match)
        else:
            others.append(match)

    timeline: list[tuple[dt.date, str, Any]] = [
        (min(group, key=lambda m: m.match_date).match_date, "gameweek", group)
        for group in blocks.values()
    ]
    timeline.extend((match.match_date, "other", match) for match in others)
    # Gameweeks are ordered before other fixtures sharing a date so that a
    # midweek cup tie does not leak into the weekend's features.
    timeline.sort(key=lambda item: (item[0], 0 if item[1] == "gameweek" else 1))
    return timeline


def _apply_result(
    match: MatchRow,
    states: dict[str, TeamState],
    elo: EloTable,
    head_to_head: HeadToHead,
) -> None:
    if match.home_goals is None or match.away_goals is None:
        return
    is_league = match.competition in _LEAGUE_COMPETITIONS
    home_elo = elo.rating(match.home_team, match.competition)
    away_elo = elo.rating(match.away_team, match.competition)

    states[match.home_team].record(
        MatchOutcome(
            date=match.match_date,
            competition=match.competition,
            is_home=True,
            goals_for=match.home_goals,
            goals_against=match.away_goals,
            shots_for=match.stats.get("home_shots"),
            shots_against=match.stats.get("away_shots"),
            shots_on_target_for=match.stats.get("home_shots_on_target"),
            shots_on_target_against=match.stats.get("away_shots_on_target"),
            opponent=match.away_team,
            opponent_elo=away_elo,
        ),
        is_league=is_league,
    )
    states[match.away_team].record(
        MatchOutcome(
            date=match.match_date,
            competition=match.competition,
            is_home=False,
            goals_for=match.away_goals,
            goals_against=match.home_goals,
            shots_for=match.stats.get("away_shots"),
            shots_against=match.stats.get("home_shots"),
            shots_on_target_for=match.stats.get("away_shots_on_target"),
            shots_on_target_against=match.stats.get("home_shots_on_target"),
            opponent=match.home_team,
            opponent_elo=home_elo,
        ),
        is_league=is_league,
    )
    head_to_head.record(
        match.home_team,
        match.away_team,
        match.match_date,
        match.home_goals,
        match.away_goals,
        match.competition,
    )
    elo.update(
        match.home_team,
        match.away_team,
        match.home_goals,
        match.away_goals,
        weight=COMPETITION_WEIGHTS.get(match.competition, 0.5),
    )


class FeatureBuilder:
    """Walks the fixture list once, emitting one feature row per match."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.matches = load_matches(conn)
        self.managers = _load_managers(conn)
        self.squad_ratings = _load_squad_ratings(conn)
        self.rating_scale = _season_z_scores(self.squad_ratings, "overall")
        self.unavailability = _load_unavailability(conn)
        self.european = _load_european(conn)
        self.settings = config.FEATURES

        self.states: dict[str, TeamState] = defaultdict(TeamState)
        self.elo = EloTable(self.settings)
        self.head_to_head = HeadToHead()

        # Fixture list lookups that are legitimately knowable in advance:
        # the schedule is published months ahead, so "when does this club
        # play next" is not hindsight.
        self.future_dates: dict[str, list[dt.date]] = defaultdict(list)
        for match in self.matches:
            self.future_dates[match.home_team].append(match.match_date)
            self.future_dates[match.away_team].append(match.match_date)
        for dates in self.future_dates.values():
            dates.sort()

        self.league_members: dict[str, set[str]] = defaultdict(set)
        for match in self.matches:
            if match.competition == config.TARGET_COMPETITION:
                self.league_members[match.season].add(match.home_team)
                self.league_members[match.season].add(match.away_team)

        self.season_order = sorted(self.league_members)
        self.matchdays_per_season = {
            season: max(
                (m.matchday or 0)
                for m in self.matches
                if m.season == season and m.competition == config.TARGET_COMPETITION
            )
            for season in self.season_order
        }

    # -- individual feature families ------------------------------------

    def _next_match_date(self, team: str, after: dt.date) -> dt.date | None:
        for date in self.future_dates.get(team, ()):
            if date > after:
                return date
        return None

    def _congestion(self, team: str, date: dt.date, season: str) -> dict[str, float | None]:
        state = self.states[team]
        next_date = self._next_match_date(team, date)
        european = self.european.get((season, team))
        return {
            "rest_days": state.days_since_last_match(date, self.settings.max_rest_days),
            "matches_last_14d": float(state.matches_within(date, 14)),
            "matches_last_21d": float(state.matches_within(date, 21)),
            "days_to_next_match": (
                min((next_date - date).days, self.settings.max_rest_days)
                if next_date
                else None
            ),
            "is_midweek": 1.0 if date.weekday() in (1, 2, 3) else 0.0,
            "european_tier": _EUROPEAN_TIERS.get(european or "", 0.0),
        }

    def _manager(self, team: str, date: dt.date) -> dict[str, float | None]:
        """Tenure and the change in results either side of an appointment.

        "New manager bounce" is not something that can be looked up, so it
        is expressed as the things that can: how long the manager has been
        there, and how the club's results under them compare with the
        results under the person before. The model decides whether either
        matters.
        """
        spells = self.managers.get(team)
        if not spells:
            return {
                "manager_days": None,
                "manager_matches": None,
                "manager_is_new": None,
                "manager_ppg_delta": None,
            }
        current = None
        for spell in spells:
            if spell["start"] <= date and (spell["end"] is None or date <= spell["end"]):
                current = spell
        if current is None:
            return {
                "manager_days": None,
                "manager_matches": None,
                "manager_is_new": None,
                "manager_ppg_delta": None,
            }

        recent = list(self.states[team].recent)
        under_current = [game for game in recent if game.date >= current["start"]]
        under_previous = [game for game in recent if game.date < current["start"]]

        def ppg(games: list[MatchOutcome]) -> float | None:
            return sum(game.points for game in games) / len(games) if games else None

        current_ppg, previous_ppg = ppg(under_current), ppg(under_previous)
        days = (date - current["start"]).days
        return {
            "manager_days": float(min(days, 2000)),
            "manager_matches": float(len(under_current)),
            "manager_is_new": 1.0 if len(under_current) < 6 else 0.0,
            "manager_ppg_delta": (
                current_ppg - previous_ppg
                if current_ppg is not None and previous_ppg is not None
                else None
            ),
        }

    def _squad(self, team: str, season: str) -> dict[str, float | None]:
        rating = self.squad_ratings.get((season, team))
        if rating is None:
            return {
                "squad_overall_z": None,
                "squad_attack": None,
                "squad_defence": None,
                "squad_rating_age": None,
            }
        mean, spread = self.rating_scale.get(rating["season"], (None, None))
        overall = rating.get("overall")
        return {
            "squad_overall_z": (
                (float(overall) - mean) / spread
                if overall is not None and mean is not None
                else None
            ),
            "squad_attack": rating.get("attack"),
            "squad_defence": rating.get("defence"),
            "squad_rating_age": rating.get("rating_age_seasons"),
        }

    def _availability(self, season: str, matchday: int | None, team: str) -> dict[str, float | None]:
        row = self.unavailability.get((season, matchday, team)) if matchday else None
        if row is None:
            return {"players_out": None, "key_players_out": None}
        return {
            "players_out": float(row["players_out"]),
            "key_players_out": float(row["key_players_out"]),
        }

    def _table_context(
        self, table: dict[str, dict[str, float]], team: str, size: int
    ) -> dict[str, float | None]:
        """Where a club sits, and what it is currently playing for.

        Position alone is a weak signal; the distance to the outcomes a
        club is actually chasing or fearing is the thing that changes how
        a season's remaining matches are approached.
        """
        row = table.get(team)
        if row is None or not table:
            return {
                "table_position": None,
                "table_played": None,
                "table_points": None,
                "table_ppg": None,
                "points_off_top": None,
                "points_off_top4": None,
                "points_off_top6": None,
                "points_above_drop": None,
            }
        ordered = sorted(table.values(), key=lambda item: item["position"])

        def points_at(position: int) -> float | None:
            index = position - 1
            return float(ordered[index]["points"]) if 0 <= index < len(ordered) else None

        points = float(row["points"])
        played = max(int(row["played"]), 1)
        drop_zone_top = points_at(size - 2)  # first club inside the bottom three
        return {
            "table_position": float(row["position"]),
            "table_played": float(row["played"]),
            "table_points": points,
            "table_ppg": points / played if row["played"] else None,
            "points_off_top": (points_at(1) or points) - points,
            "points_off_top4": (
                (points_at(4) - points) if points_at(4) is not None else None
            ),
            "points_off_top6": (
                (points_at(6) - points) if points_at(6) is not None else None
            ),
            "points_above_drop": (
                (points - drop_zone_top) if drop_zone_top is not None else None
            ),
        }

    def _team_block(
        self,
        team: str,
        season: str,
        matchday: int | None,
        date: dt.date,
        table: dict[str, dict[str, float]],
        size: int,
        *,
        is_home: bool,
    ) -> dict[str, float | None]:
        state = self.states[team]
        block: dict[str, float | None] = {"elo": self.elo.rating(team, config.TARGET_COMPETITION)}

        for window in self.settings.form_windows:
            for name, value in state.form(
                window, today=date, max_age_days=self.settings.form_max_age_days
            ).items():
                block[f"form{window}_{name}"] = value

        if is_home:
            block["venue_ppg"] = (
                state.home_points / state.home_played if state.home_played else None
            )
            block["venue_goals_for"] = (
                state.home_goals_for / state.home_played if state.home_played else None
            )
            block["venue_goals_against"] = (
                state.home_goals_against / state.home_played if state.home_played else None
            )
        else:
            block["venue_ppg"] = (
                state.away_points / state.away_played if state.away_played else None
            )
            block["venue_goals_for"] = (
                state.away_goals_for / state.away_played if state.away_played else None
            )
            block["venue_goals_against"] = (
                state.away_goals_against / state.away_played if state.away_played else None
            )

        block["previous_season_ppg"] = state.previous_ppg
        block["previous_season_gd_per_game"] = state.previous_goal_difference_per_game
        # The previous season may have been played in a lower division,
        # where points come more easily, so the model is told which.
        block["previous_season_was_top_flight"] = (
            1.0 if state.previous_competition == config.TARGET_COMPETITION else 0.0
        )
        block["seasons_tracked"] = float(state.seasons_seen)
        block["top_flight_seasons"] = float(state.top_flight_seasons)
        block["is_promoted"] = (
            1.0
            if season in self.season_order
            and team not in self.league_members.get(self._previous_season(season), set())
            else 0.0
        )

        block.update(self._table_context(table, team, size))
        block.update(self._congestion(team, date, season))
        block.update(self._manager(team, date))
        block.update(self._squad(team, season))
        block.update(self._availability(season, matchday, team))
        return block

    def _previous_season(self, season: str) -> str:
        index = self.season_order.index(season) if season in self.season_order else 0
        return self.season_order[index - 1] if index > 0 else season

    # -- the pass itself -------------------------------------------------

    def build(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, kind, payload in _gameweek_blocks(self.matches):
            if kind == "other":
                self._prepare_teams(payload)
                _apply_result(payload, self.states, self.elo, self.head_to_head)
                continue

            group: list[MatchRow] = payload
            for match in group:
                self._prepare_teams(match)
            season = group[0].season
            members = self.league_members[season]
            table = LeagueTable(self.states).snapshot(members)
            for match in group:
                rows.append(self._featurise(match, table, len(members)))
            for match in group:
                _apply_result(match, self.states, self.elo, self.head_to_head)

        frame = pd.DataFrame(rows)
        return frame.sort_values(["season", "matchday", "match_date"]).reset_index(drop=True)

    def _prepare_teams(self, match: MatchRow) -> None:
        for team in (match.home_team, match.away_team):
            self.elo.start_season(team, match.season, match.competition)
            if match.competition in _LEAGUE_COMPETITIONS:
                self.states[team].roll_season(match.season, match.competition)

    def _featurise(
        self, match: MatchRow, table: dict[str, dict[str, float]], size: int
    ) -> dict[str, Any]:
        date = match.match_date
        home = self._team_block(
            match.home_team, match.season, match.matchday, date, table, size, is_home=True
        )
        away = self._team_block(
            match.away_team, match.season, match.matchday, date, table, size, is_home=False
        )

        row: dict[str, Any] = {
            "match_id": match.match_id,
            "season": match.season,
            "matchday": match.matchday,
            "match_date": date.isoformat(),
            "home_team": match.home_team,
            "away_team": match.away_team,
            "status": match.status,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
        }
        for name, value in home.items():
            row[f"home_{name}"] = value
        for name, value in away.items():
            row[f"away_{name}"] = value

        # Differences carry most of the signal, and giving them to the
        # model directly saves it from having to learn subtraction.
        for name in home:
            left, right = home[name], away[name]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                row[f"diff_{name}"] = float(left) - float(right)
            else:
                row[f"diff_{name}"] = None

        row["elo_expected_home"] = self.elo.expected_home_score(
            match.home_team, match.away_team
        )
        row.update(self.head_to_head.summary(match.home_team, match.away_team, date))

        total_matchdays = self.matchdays_per_season.get(match.season, 38) or 38
        row["season_progress"] = (match.matchday or 0) / total_matchdays
        row["matchday_number"] = float(match.matchday or 0)
        row["matchdays_remaining"] = float(total_matchdays - (match.matchday or 0))
        row["is_weekend"] = 1.0 if date.weekday() >= 5 else 0.0

        if match.home_goals is not None and match.away_goals is not None:
            row["result"] = (
                "H"
                if match.home_goals > match.away_goals
                else ("A" if match.home_goals < match.away_goals else "D")
            )
        else:
            row["result"] = None
        return row


META_COLUMNS = (
    "match_id",
    "season",
    "matchday",
    "match_date",
    "home_team",
    "away_team",
    "status",
    "home_goals",
    "away_goals",
    "result",
)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_features(db_path: str | None = None) -> pd.DataFrame:
    with db.connect(db_path) as conn:
        return FeatureBuilder(conn).build()


def store_features(frame: pd.DataFrame, db_path: str | None = None) -> int:
    """Persist the feature table, one JSON payload per match."""
    built_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    columns = feature_columns(frame)
    rows = [
        {
            "match_id": record["match_id"],
            "season": record["season"],
            "matchday": record["matchday"],
            "match_date": record["match_date"],
            "feature_version": FEATURE_VERSION,
            "built_at": built_at,
            "payload": db.dumps(
                {
                    column: (None if pd.isna(record[column]) else float(record[column]))
                    for column in columns
                }
            ),
        }
        for record in frame.to_dict("records")
    ]
    with db.connect(db_path) as conn:
        return db.upsert_many(conn, "features", rows, ("match_id",))
