"""Reader for the openfootball/england plain-text match archive.

openfootball publishes one text file per competition per season, with
results committed within a day or two of each round being played. It is
free, has no API key, covers every English division plus the domestic
cups, and — crucially for a gameweek-oriented platform — labels each
block of fixtures with its matchday number.

The archive has drifted between two layouts over the years, so the
parser accepts both:

    Matchday-first, "v" separated (most seasons)
        20:00  Arsenal FC  v Coventry City FC  3-0 (2-0)

    Score-in-the-middle (older seasons and 2025/26)
        12:45  Manchester United  1-0 (1-0)  Tottenham Hotspur

Unplayed fixtures simply omit the score, which is how the current
gameweek's matches arrive.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# "▪ Matchday 12", "▪ Regular Season - 12", "▪ Round 3"
_ROUND_RE = re.compile(
    r"^[▪\-*]\s*(?:Matchday|Match Day|Regular Season\s*-|Round|Week)\s*(\d+)",
    re.IGNORECASE,
)
# A stage heading with no number, e.g. "▪ Final" or "▪ Semi-finals".
_STAGE_RE = re.compile(r"^[▪\-*]\s*(.+?)\s*$")

# "Sat Aug 16 2024" or "Sat Aug 16"
_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+"
    r"(?:(\d{1,2})\s+([A-Z][a-z]{2})|([A-Z][a-z]{2})\s+(\d{1,2}))"
    r"(?:\s+(\d{4}))?\s*$"
)

_TIME_RE = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s+(.*)$")

# 3-0 (2-0)  |  3-0
_SCORE = r"(\d{1,2})\s*[-\u2013]\s*(\d{1,2})"
_HT = r"(?:\s*\(\s*(\d{1,2})\s*[-\u2013]\s*(\d{1,2})\s*\))?"

_MATCH_V_RE = re.compile(
    rf"^(?P<home>.+?)\s+v\.?\s+(?P<away>.+?)(?:\s+{_SCORE}{_HT})?\s*$",
    re.IGNORECASE,
)
_MATCH_MID_RE = re.compile(
    rf"^(?P<home>.+?)\s+{_SCORE}{_HT}\s+(?P<away>.+?)\s*$"
)

# Knockout ties carry the decisive score first and the regulation score
# in brackets:  "10-9 pen. (1-1, 0-1)"  or  "2-6 a.e.t. (2-2, 0-2)".
# The bracketed pairs are the 90-minute and half-time scores, which are
# the ones comparable with a league result, so the line is rewritten to
# the ordinary "full-time (half-time)" shape before it is parsed.
_EXTRA_TIME_RE = re.compile(
    r"\s+(?:\d{1,2}\s*[-\u2013]\s*\d{1,2}\s*"
    r"(?:pens?\.?|a\.e\.t\.|aet)\.?\s*)+"
    r"\(\s*(?P<regulation>\d{1,2}\s*[-\u2013]\s*\d{1,2})\s*"
    r"(?:,\s*(?P<halftime>\d{1,2}\s*[-\u2013]\s*\d{1,2})\s*)?\)",
    re.IGNORECASE,
)

# Bracketed editorial notes: "[awarded]", "[abandoned]".
_ANNOTATION_RE = re.compile(r"\s*\[(?P<note>[^\]]*)\]")


def _normalise_result_notation(line: str) -> tuple[str, bool]:
    """Flatten knockout notation and strip editorial annotations.

    Returns the rewritten line plus a flag saying whether the fixture
    should be kept at all (abandoned matches never happened).
    """
    keep = True

    def _drop_annotation(found: re.Match[str]) -> str:
        nonlocal keep
        if "abandon" in found.group("note").lower():
            keep = False
        return " "

    line = _ANNOTATION_RE.sub(_drop_annotation, line)
    line = _EXTRA_TIME_RE.sub(
        lambda m: f"  {m.group('regulation')}"
        + (f" ({m.group('halftime')})" if m.group("halftime") else ""),
        line,
    )
    return line.strip(), keep

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}

# Lines that are neither fixtures nor headings.
_NOISE_PREFIXES = ("#", "=", "//")


@dataclass(frozen=True)
class RawMatch:
    """One fixture exactly as the archive states it, before name mapping."""

    season: str
    competition: str
    matchday: int | None
    stage: str | None
    date: dt.date | None
    kickoff: str | None
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    ht_home_goals: int | None
    ht_away_goals: int | None

    @property
    def is_played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


def _season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def _clean_team(name: str) -> str:
    """Strip the decorative bits openfootball appends to club names."""
    name = name.strip().strip("-").strip()
    # Drop trailing annotations such as "(1)" for aggregate scores.
    name = re.sub(r"\s*\((?:\d+|agg[^)]*)\)\s*$", "", name)
    return re.sub(r"\s{2,}", " ", name).strip()


def _looks_like_team(text: str) -> bool:
    """Reject goal-scorer lines, lineups and other prose."""
    if not text or len(text) > 60:
        return False
    if text.startswith(("(", "[", "Yellow", "Red", "Att:")):
        return False
    # Scorer continuation lines are full of apostrophes and semicolons.
    return not any(ch in text for ch in "';")


class SeasonFileParser:
    """Parses a single openfootball competition file."""

    def __init__(self, season: str, competition: str) -> None:
        self.season = season
        self.competition = competition
        self._start_year = _season_start_year(season)
        self._matchday: int | None = None
        self._stage: str | None = None
        self._date: dt.date | None = None
        self._kickoff: str | None = None
        self._last_month: int | None = None
        self._year: int = self._start_year

    def parse(self, text: str) -> list[RawMatch]:
        matches: list[RawMatch] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(_NOISE_PREFIXES):
                continue
            if self._consume_heading(stripped):
                continue
            if self._consume_date(stripped):
                continue
            body = self._consume_time(stripped)
            match = self._parse_fixture(body)
            if match is not None:
                matches.append(match)
        return matches

    # -- heading / date / time state ------------------------------------

    def _consume_heading(self, line: str) -> bool:
        if not line.startswith(("▪", "»")):
            return False
        round_match = _ROUND_RE.match(line)
        if round_match:
            self._matchday = int(round_match.group(1))
            self._stage = None
        else:
            stage = _STAGE_RE.match(line)
            self._matchday = None
            self._stage = stage.group(1).lstrip("▪» ").strip() if stage else None
        # A new block resets the kickoff time but not the date, since
        # openfootball repeats the date only when it changes.
        self._kickoff = None
        return True

    def _consume_date(self, line: str) -> bool:
        found = _DATE_RE.match(line)
        if not found:
            return False
        day_a, mon_a, mon_b, day_b, year = found.groups()
        day = int(day_a or day_b)
        month = _MONTHS[(mon_a or mon_b)]
        if year:
            self._year = int(year)
        elif self._last_month is not None and month < self._last_month:
            # The calendar year rolls over inside a season: a month that
            # goes backwards means we have crossed into January.
            self._year += 1
        self._last_month = month
        try:
            self._date = dt.date(self._year, month, day)
        except ValueError:
            self._date = None
        self._kickoff = None
        return True

    def _consume_time(self, line: str) -> str:
        found = _TIME_RE.match(line)
        if not found:
            return line
        hour, minute, rest = found.groups()
        self._kickoff = f"{int(hour):02d}:{minute}"
        return rest.strip()

    # -- fixtures --------------------------------------------------------

    def _parse_fixture(self, body: str) -> RawMatch | None:
        if not body or body.startswith(_NOISE_PREFIXES):
            return None

        body, keep = _normalise_result_notation(body)
        if not keep or not body:
            return None

        found = _MATCH_V_RE.match(body)
        if found:
            groups = found.groupdict()
            home, away = _clean_team(groups["home"]), _clean_team(groups["away"])
            scores = found.groups()[2:6]
        else:
            found = _MATCH_MID_RE.match(body)
            if not found:
                return None
            groups = found.groupdict()
            home, away = _clean_team(groups["home"]), _clean_team(groups["away"])
            scores = found.groups()[1:5]

        if not (_looks_like_team(home) and _looks_like_team(away)) or home == away:
            return None

        full_home, full_away, ht_home, ht_away = (
            int(value) if value is not None else None for value in scores
        )
        return RawMatch(
            season=self.season,
            competition=self.competition,
            matchday=self._matchday,
            stage=self._stage,
            date=self._date,
            kickoff=self._kickoff,
            home_team=home,
            away_team=away,
            home_goals=full_home,
            away_goals=full_away,
            ht_home_goals=ht_home,
            ht_away_goals=ht_away,
        )


def parse_file(path: Path, season: str, competition: str) -> list[RawMatch]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return SeasonFileParser(season, competition).parse(text)


def sync_checkout(checkout: Path, repo_url: str) -> Path:
    """Clone the archive on first run, then fast-forward it on later runs.

    Keeping a real checkout rather than fetching individual files means
    the nightly job pulls only the handful of lines that changed since
    the last gameweek.
    """
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


def available_seasons(checkout: Path) -> list[str]:
    pattern = re.compile(r"^\d{4}-\d{2}$")
    return sorted(p.name for p in Path(checkout).iterdir()
                  if p.is_dir() and pattern.match(p.name))


def read_season(
    checkout: Path, season: str, competition_files: dict[str, str]
) -> list[RawMatch]:
    """Read every available competition file for one season."""
    season_dir = Path(checkout) / season
    matches: list[RawMatch] = []
    if not season_dir.is_dir():
        return matches
    for competition, filename in competition_files.items():
        path = season_dir / filename
        if path.is_file():
            matches.extend(parse_file(path, season, competition))
    return matches
