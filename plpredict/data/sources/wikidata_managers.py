"""Managerial spells from Wikidata.

The manager features — how long the current manager has been in post,
how the club has done under them against the one before — need every
club's spells with their dates, back through the training window. A
hand-kept file only ever covered the most recent seasons, which left
the model a couple of seasons of evidence to learn "new manager bounce"
from.

Wikidata records the same facts as Wikipedia, as data rather than
prose, from two directions: a club's "head coach" statements and a
manager's "coach of sports team" statements, each with start and end
dates. Either is often missing where the other is present, so both are
read. It is maintained by the same editors who update Wikipedia within
hours of an appointment, and it can be queried in one request, which
makes it both the historical backfill and the daily source of changes.

Clubs are matched to their Wikidata items once and pinned in
``data/manual/team_wikidata.csv``. A club not yet pinned — a newly
promoted side — is resolved by name at sync time.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from plpredict.data.teams import canonical_name

SPARQL_URL = "https://query.wikidata.org/sparql"
_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "plpredict/1.0 (football research project)",
}
# Roles that mean being in charge of the team. Assistant and goalkeeping
# coaches are recorded with the same "coach of sports team" property and
# are told apart only by a role like these.
_IN_CHARGE = ", ".join(
    f"wd:{item}"
    for item in (
        "Q3246315",  # head coach
        "Q1050607",  # caretaker manager
        "Q4895105",  # interim
        "Q1413682",  # interim management
        "Q379533",  # player-coach
    )
)

# A spell recorded from both directions often disagrees by a few days
# (announcement against first day in post). Within this many days it is
# the same spell.
_SAME_SPELL_DAYS = 45


@dataclass(frozen=True)
class Spell:
    team: str
    manager: str
    start_date: str
    end_date: str | None


def _query(sparql: str, timeout: float = 180.0) -> list[dict[str, Any]]:
    response = requests.post(SPARQL_URL, data={"query": sparql}, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def read_club_ids(path: Path) -> dict[str, str]:
    if not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    return {row["team"]: row["wikidata_id"] for row in csv.DictReader(lines)}


def resolve_club_ids(teams: list[str]) -> dict[str, str]:
    """Find clubs' Wikidata items by name.

    Several items can carry a club's name — a women's side, a defunct
    namesake, a club abroad. The one with the most head-coach records is
    the senior club, which is the one wanted.
    """
    labels = set()
    for team in teams:
        labels |= {team, f"{team} F.C.", f"{team} A.F.C.", f"AFC {team}", f"{team} FC"}
    values = " ".join(f'"{label}"@en' for label in sorted(labels))
    rows = _query(
        f"""SELECT ?club ?label (COUNT(?st) AS ?n) WHERE {{
              VALUES ?label {{ {values} }}
              ?club rdfs:label ?label ; p:P286 ?st .
            }} GROUP BY ?club ?label"""
    )
    best: dict[str, tuple[int, str]] = {}
    for row in rows:
        team = canonical_name(row["label"]["value"])
        count = int(row["n"]["value"])
        item = row["club"]["value"].rsplit("/", 1)[1]
        if team in teams and (team not in best or count > best[team][0]):
            best[team] = (count, item)
    return {team: item for team, (_, item) in best.items()}


def fetch_spells(club_ids: dict[str, str]) -> list[Spell]:
    by_item = {item: team for team, item in club_ids.items()}
    values = " ".join(f"wd:{item}" for item in by_item)
    rows = _query(
        f"""SELECT ?club ?coachLabel ?start ?startPrecision ?end WHERE {{
              VALUES ?club {{ {values} }}
              {{ ?club p:P286 ?st . ?st ps:P286 ?coach . }}
              UNION
              {{ ?coach p:P6087 ?st . ?st ps:P6087 ?club . }}
              # "Coach of sports team" covers the whole staff; a statement
              # naming any role other than the one in charge is dropped.
              FILTER NOT EXISTS {{
                ?st pq:P39|pq:P2868|pq:P3831 ?role .
                FILTER(?role NOT IN ({_IN_CHARGE}))
              }}
              OPTIONAL {{ ?st pqv:P580 ?startValue .
                         ?startValue wikibase:timeValue ?start ; wikibase:timePrecision ?startPrecision }}
              OPTIONAL {{ ?st pq:P582 ?end }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}"""
    )
    raw: list[tuple[Spell, bool]] = []
    for row in rows:
        start = row.get("start", {}).get("value", "")[:10]
        if not start:
            continue  # a spell with no start date cannot be placed
        end = row.get("end", {}).get("value", "")[:10] or None
        spell = Spell(
            team=by_item[row["club"]["value"].rsplit("/", 1)[1]],
            manager=row["coachLabel"]["value"],
            start_date=start,
            end_date=end,
        )
        # Precision 9 is a year: "2022" is stored as 1 January 2022.
        year_only = int(row.get("startPrecision", {}).get("value", 11)) <= 9
        raw.append((spell, year_only))
    return tidy(drop_imprecise(raw))


def _overlaps(a: Spell, b: Spell) -> bool:
    a_end = a.end_date or "9999-12-31"
    b_end = b.end_date or "9999-12-31"
    return a.start_date <= b_end and b.start_date <= a_end


def drop_imprecise(raw: list[tuple[Spell, bool]]) -> list[Spell]:
    """Discard a year-only record where an exact one describes the same spell.

    A year-only start reads as 1 January, so beside the exact record of
    the same appointment it looks like a second, earlier spell. Where no
    exact record exists it is the best available and is kept.
    """
    exact = [spell for spell, year_only in raw if not year_only]
    kept = list(exact)
    for spell, year_only in raw:
        if not year_only:
            continue
        duplicate = any(
            other.team == spell.team
            and other.manager == spell.manager
            and (other.start_date[:4] == spell.start_date[:4] or _overlaps(other, spell))
            for other in exact
        )
        if not duplicate:
            kept.append(spell)
    return kept


def tidy(spells: list[Spell]) -> list[Spell]:
    """One clean, ordered timeline per club.

    The same spell recorded from both directions is merged, keeping the
    earlier start and the later end. An open-ended spell followed by
    another manager's is closed the day before the next one starts:
    Wikidata often records an appointment without anyone going back to
    end the previous spell.
    """
    by_team: dict[str, list[Spell]] = defaultdict(list)
    for spell in spells:
        by_team[spell.team].append(spell)

    tidied: list[Spell] = []
    for team, group in by_team.items():
        group.sort(key=lambda s: (s.start_date, s.manager))
        merged: list[Spell] = []
        for spell in group:
            same = next(
                (
                    m
                    for m in merged
                    if m.manager == spell.manager
                    and abs(
                        (dt.date.fromisoformat(m.start_date) - dt.date.fromisoformat(spell.start_date)).days
                    )
                    <= _SAME_SPELL_DAYS
                ),
                None,
            )
            if same is None:
                merged.append(spell)
                continue
            ends = [e for e in (same.end_date, spell.end_date) if e]
            merged[merged.index(same)] = Spell(
                team, same.manager, min(same.start_date, spell.start_date), max(ends) if ends else None
            )

        merged.sort(key=lambda s: s.start_date)
        for index, spell in enumerate(merged):
            later = [m for m in merged[index + 1 :] if m.manager != spell.manager]
            if spell.end_date is None and later:
                day_before = dt.date.fromisoformat(later[0].start_date) - dt.timedelta(days=1)
                spell = Spell(team, spell.manager, spell.start_date, day_before.isoformat())
            tidied.append(spell)
    return sorted(tidied, key=lambda s: (s.team, s.start_date, s.manager))


COLUMNS = ("team", "manager", "start_date", "end_date")


def write(spells: list[Spell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for spell in spells:
            writer.writerow((spell.team, spell.manager, spell.start_date, spell.end_date or ""))


def read(path: Path) -> list[Spell]:
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            Spell(row["team"], row["manager"], row["start_date"], row["end_date"] or None)
            for row in csv.DictReader(handle)
        ]


def coverage(spells: list[Spell], team_dates: list[tuple[str, str]]) -> float:
    """Share of (club, match date) pairs that fall inside a known spell."""
    by_team: dict[str, list[Spell]] = defaultdict(list)
    for spell in spells:
        by_team[spell.team].append(spell)
    if not team_dates:
        return 1.0
    covered = sum(
        1
        for team, date in team_dates
        if any(
            s.start_date <= date and (s.end_date is None or date <= s.end_date)
            for s in by_team.get(team, ())
        )
    )
    return covered / len(team_dates)
