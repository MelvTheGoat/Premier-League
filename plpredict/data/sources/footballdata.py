"""Match statistics from the football-data.co.uk archive.

openfootball supplies the fixture list and the scoreline; this source
supplies what happened inside the match — shots, shots on target,
corners, fouls, cards and the referee. Shots on target is the closest
freely available stand-in for expected goals, and a team's rolling
shot-creation and shot-concession rates are a better read on underlying
performance than goals alone, which are noisy in small samples.

The archive is consumed through the ``datasets/football-datasets``
mirror, which republishes the same CSVs on a schedule with stable
column names.

This source is deliberately optional. It lags the live season, so the
feature layer marks the columns it produces as an enrichment: any of
them that is unavailable for the gameweek being predicted is dropped
from that run's model rather than served as a missing value.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plpredict.data.teams import canonical_name

# football-data's column codes, mapped to names the rest of the code uses.
STAT_COLUMNS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

_SEASON_FILE_RE = re.compile(r"season-(\d{2})(\d{2})\.csv$")


@dataclass(frozen=True)
class MatchStats:
    season: str
    match_date: dt.date
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    referee: str | None
    stats: dict[str, float]


def season_from_filename(path: Path) -> str | None:
    """"season-2526.csv" -> "2025-26"."""
    found = _SEASON_FILE_RE.search(Path(path).name)
    if not found:
        return None
    start_code = int(found.group(1))
    start_year = 1900 + start_code if start_code >= 90 else 2000 + start_code
    return f"{start_year}-{found.group(2)}"


def sync_checkout(checkout: Path, repo_url: str) -> Path:
    checkout = Path(checkout)
    if (checkout / ".git").exists():
        # A failed pull is not fatal: the checkout on disk is still
        # perfectly good data, just missing the newest results. Falling
        # over here would take the whole pipeline down over a network
        # blip or a local edit to the checkout.
        result = subprocess.run(
            ["git", "-C", str(checkout), "pull", "--ff-only", "--quiet"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(
                f"warning: could not refresh {checkout}: "
                f"{result.stderr.strip() or 'git pull failed'}"
            )
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo_url, str(checkout)],
            check=True,
            timeout=900,
        )
    return checkout


def _parse_date(value: object) -> dt.date | None:
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
            try:
                return dt.datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def read_season_file(path: Path, season: str | None = None) -> list[MatchStats]:
    season = season or season_from_filename(path)
    if season is None:
        raise ValueError(f"Cannot infer a season from {path}")

    frame = pd.read_csv(path, low_memory=False)
    if "HomeTeam" not in frame or "AwayTeam" not in frame:
        return []

    records: list[MatchStats] = []
    for row in frame.to_dict("records"):
        date = _parse_date(row.get("Date"))
        home = canonical_name(str(row.get("HomeTeam") or ""))
        away = canonical_name(str(row.get("AwayTeam") or ""))
        if not (date and home and away):
            continue
        stats = {}
        for code, name in STAT_COLUMNS.items():
            value = pd.to_numeric(row.get(code), errors="coerce")
            if pd.notna(value):
                stats[name] = float(value)
        referee = row.get("Referee")
        records.append(
            MatchStats(
                season=season,
                match_date=date,
                home_team=home,
                away_team=away,
                home_goals=_as_int(row.get("FTHG")),
                away_goals=_as_int(row.get("FTAG")),
                referee=str(referee).strip() if isinstance(referee, str) else None,
                stats=stats,
            )
        )
    return records


def _as_int(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    return int(number) if pd.notna(number) else None


def read_all(
    checkout: Path,
    subdirectory: str = "datasets/premier-league",
    first_season: str | None = None,
) -> list[MatchStats]:
    directory = Path(checkout) / subdirectory
    if not directory.is_dir():
        return []
    records: list[MatchStats] = []
    for path in sorted(directory.glob("season-*.csv")):
        season = season_from_filename(path)
        if season is None or (first_season and season < first_season):
            continue
        records.extend(read_season_file(path, season))
    return records
