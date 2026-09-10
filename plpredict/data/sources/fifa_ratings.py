"""Squad-quality proxy built from FIFA / EA FC player ratings.

Why a video game: EA rate every professional squad once a season, on a
consistent scale, going back two decades. That makes their per-player
overall rating the only freely available, historically complete measure
of how good a squad is *on paper* — which is exactly the thing league
tables cannot tell you about a newly promoted side or a club that has
just spent heavily.

Ratings are aggregated to one row per club per season here rather than
carried around at player level, because the model wants squad strength,
not a player database.

Two source layouts are understood:

``fifa_model``  lbenz730/fifa_model — ``rating``, ``club``,
                ``preferred_positions``, one file per game year.
``sofifa``      the widely mirrored sofifa exports (``players_22.csv``,
                EA FC data hubs) — ``overall``, ``club_name``,
                ``player_positions``, ``league_name``.

A game version maps to the season it was released into: FIFA 20 shipped
in September 2019 and therefore describes squads for 2019/20.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plpredict.data.teams import canonical_name

GK = "GK"
DEF = "DEF"
MID = "MID"
ATT = "ATT"

_POSITION_GROUPS = {
    GK: {"GK"},
    DEF: {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB", "SW", "RCB", "LCB"},
    MID: {"CDM", "LDM", "RDM", "CM", "LCM", "RCM", "CAM", "LAM", "RAM", "LM", "RM"},
    ATT: {"ST", "LS", "RS", "CF", "LF", "RF", "LW", "RW"},
}
_POSITION_LOOKUP = {
    position: group for group, positions in _POSITION_GROUPS.items() for position in positions
}

# How many players each aggregate averages over. Squad depth matters,
# but a club's best XI matters more, so the headline number is a
# depth-weighted blend rather than a flat squad mean.
_TOP_11 = 11
_SQUAD_DEPTH = 20
_UNIT_DEPTH = 5


@dataclass(frozen=True)
class SquadRating:
    season: str
    team: str
    overall: float
    attack: float | None
    midfield: float | None
    defence: float | None
    top11_mean: float
    squad_value: float | None
    source: str


def season_for_version(version: int) -> str:
    """FIFA/FC 20 -> "2019-20": the game ships at the start of the season."""
    start = 2000 + version - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _primary_position(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    first = re.split(r"[/,]", raw.strip())[0].strip().upper()
    return _POSITION_LOOKUP.get(first)


def _unit_mean(frame: pd.DataFrame, group: str) -> float | None:
    unit = frame.loc[frame["position_group"] == group, "rating"]
    if unit.empty:
        return None
    return float(unit.nlargest(_UNIT_DEPTH).mean())


def _aggregate(frame: pd.DataFrame, season: str, source: str) -> list[SquadRating]:
    ratings: list[SquadRating] = []
    for team, squad in frame.groupby("team", sort=True):
        best = squad["rating"].nlargest(_TOP_11)
        if len(best) < 8:
            # Too few rated players to say anything useful about the squad.
            continue
        depth = squad["rating"].nlargest(_SQUAD_DEPTH)
        top11 = float(best.mean())
        # 70/30 best-XI to squad-depth: rotation and injuries mean depth
        # counts, but not as much as the first team.
        overall = round(0.7 * top11 + 0.3 * float(depth.mean()), 3)
        value = squad["value"].nlargest(_SQUAD_DEPTH).sum() if "value" in squad else None
        ratings.append(
            SquadRating(
                season=season,
                team=team,
                overall=overall,
                attack=_unit_mean(squad, ATT),
                midfield=_unit_mean(squad, MID),
                defence=_unit_mean(squad, DEF),
                top11_mean=round(top11, 3),
                squad_value=float(value) if value not in (None, 0) else None,
                source=source,
            )
        )
    return ratings


def _prepare(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "team": frame[columns["club"]].map(
                lambda name: canonical_name(name) if isinstance(name, str) else ""
            ),
            "rating": pd.to_numeric(frame[columns["rating"]], errors="coerce"),
            "position_group": frame[columns["positions"]].map(_primary_position),
        }
    )
    if columns.get("value") and columns["value"] in frame:
        prepared["value"] = pd.to_numeric(frame[columns["value"]], errors="coerce")
    # Loanees are rated with the club they are at, which is what we want,
    # but players with no club at all are noise.
    return prepared[(prepared["team"] != "") & prepared["rating"].notna()]


def read_fifa_model_file(path: Path) -> tuple[str, pd.DataFrame]:
    """One ``player_stats_YYYY.csv`` from lbenz730/fifa_model."""
    year = int(re.search(r"(\d{4})", Path(path).stem).group(1))
    frame = pd.read_csv(path, low_memory=False)
    columns = {
        "club": "club",
        "rating": "rating",
        "positions": "preferred_positions",
        "value": "value",
    }
    return season_for_version(year % 100), _prepare(frame, columns)


def read_sofifa_file(path: Path, version: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """A sofifa-style export, which may bundle several game versions."""
    frame = pd.read_csv(path, low_memory=False)
    columns = {
        "club": "club_name",
        "rating": "overall",
        "positions": "player_positions",
        "value": "value_eur",
    }
    missing = [name for name in ("club", "rating", "positions") if columns[name] not in frame]
    if missing:
        raise ValueError(f"{path} is not a sofifa export (missing {missing})")

    # Keep only the top division of each country; second-tier ratings are
    # loaded too so that promoted clubs have a rating in the season they
    # come up.
    if "league_level" in frame:
        frame = frame[pd.to_numeric(frame["league_level"], errors="coerce").fillna(9) <= 2]

    if "fifa_version" in frame and frame["fifa_version"].notna().any():
        groups = frame.groupby(frame["fifa_version"].astype(int))
    else:
        if version is None:
            found = re.search(r"(\d{2})", Path(path).stem)
            if not found:
                raise ValueError(f"Cannot infer a FIFA version from {path}")
            version = int(found.group(1))
        groups = [(version, frame)]

    return [
        (season_for_version(int(game_version)), _prepare(subset, columns))
        for game_version, subset in groups
    ]


def build(
    fifa_model_dir: Path | None = None,
    sofifa_files: list[Path] | None = None,
    teams_by_season: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate every source into one club-season table.

    ``teams_by_season`` restricts each season to clubs that actually
    played in England that year, which drops the foreign clubs whose
    league happens to also be called a "Premier League".
    """
    collected: list[SquadRating] = []

    if fifa_model_dir and Path(fifa_model_dir).is_dir():
        for path in sorted(Path(fifa_model_dir).glob("player_stats_*.csv")):
            season, prepared = read_fifa_model_file(path)
            collected.extend(_aggregate(prepared, season, f"fifa_model:{path.name}"))

    for path in sofifa_files or []:
        for season, prepared in read_sofifa_file(Path(path)):
            collected.extend(_aggregate(prepared, season, f"sofifa:{Path(path).name}"))

    frame = pd.DataFrame([rating.__dict__ for rating in collected])
    if frame.empty:
        return frame

    if teams_by_season:
        keep = frame.apply(
            lambda row: row["team"] in teams_by_season.get(row["season"], set()), axis=1
        )
        frame = frame[keep]

    # Later sources win when two dumps describe the same season.
    frame = frame.sort_values(["season", "team", "source"])
    frame = frame.drop_duplicates(["season", "team"], keep="last")
    return frame.sort_values(["season", "team"]).reset_index(drop=True)
