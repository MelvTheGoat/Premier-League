"""A news tripwire for managerial changes.

Wikidata is the record of who manages whom (see ``wikidata_managers``),
and it is usually updated within hours of an appointment. "Usually" is
the gap this closes. Headlines are scanned for a managerial change at a
Premier League club; if one appears and the club's record still shows
the same manager a few days later, the scheduled run says so.

Headlines are never written into the record themselves. They are far
too loose for that: "appointed" is as likely to be about a sporting
director, "leaves" about a player, and a story about a manager under
pressure reads much like one about a sacking. As a prompt to look, they
are cheap and quick; as data, they would be wrong often enough to hurt.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from plpredict.data.teams import display_name

FEED_URL = "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml"

_CHANGE = re.compile(
    r"\b(sack(ed|s|ing)?|appoint(s|ed|ment)?|names? .{0,40}(manager|head coach|boss)|"
    r"new (manager|head coach|boss)|part(s|ed)? (company|ways)|resign(s|ed)?|"
    r"(manager|head coach|boss) (leaves|departs|exits|quits)|interim (manager|head coach|boss)|"
    r"caretaker)\b",
    re.IGNORECASE,
)
# Wording that marks a story as being about someone other than the manager.
_NOT_THE_MANAGER = re.compile(
    r"\b(director|chairman|chief executive|owner|academy|under-\d+|u\d\d|women|wsl|"
    r"assistant|coach of the month|manager of the month|referee|captain|signs?|signing|"
    r"loan|transfer|contract extension)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Headline:
    title: str
    summary: str
    published: dt.date


def fetch(url: str = FEED_URL, timeout: float = 30.0) -> bytes:
    response = requests.get(url, headers={"User-Agent": "plpredict (football research project)"}, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_feed(raw: bytes) -> list[Headline]:
    headlines = []
    for item in ET.fromstring(raw).iter("item"):
        stamp = item.findtext("pubDate")
        try:
            published = email.utils.parsedate_to_datetime(stamp).date() if stamp else None
        except (TypeError, ValueError):
            published = None
        if published is None:
            continue
        headlines.append(
            Headline(
                title=(item.findtext("title") or "").strip(),
                summary=(item.findtext("description") or "").strip(),
                published=published,
            )
        )
    return headlines


# What headlines call clubs, beyond their full and short names.
_HEADLINE_NAMES = {
    "Tottenham Hotspur": ("Spurs",),
    "Manchester United": ("Man Utd", "Man United"),
    "Aston Villa": ("Villa",),
    "Nottingham Forest": ("Forest",),
    "Crystal Palace": ("Palace",),
    "Wolverhampton Wanderers": ("Wolves",),
    "Brighton & Hove Albion": ("Brighton",),
}


def _club_patterns(clubs: list[str]) -> dict[str, re.Pattern]:
    patterns = {}
    for club in clubs:
        names = {club, display_name(club), *_HEADLINE_NAMES.get(club, ())}
        alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        patterns[club] = re.compile(rf"\b({alternatives})\b", re.IGNORECASE)
    return patterns


def manager_changes(headlines: list[Headline], clubs: list[str]) -> list[tuple[str, Headline]]:
    """Headlines that read like a managerial change at one of ``clubs``."""
    patterns = _club_patterns(clubs)
    found = []
    for headline in headlines:
        text = f"{headline.title}. {headline.summary}"
        if not _CHANGE.search(headline.title) or _NOT_THE_MANAGER.search(text):
            continue
        named = [club for club, pattern in patterns.items() if pattern.search(headline.title)]
        if len(named) == 1:  # a story naming two clubs is ambiguous about which
            found.append((named[0], headline))
    return found


def unconfirmed(
    changes: list[tuple[str, Headline]],
    latest_appointment: dict[str, dt.date],
    today: dt.date,
    grace_days: int = 3,
) -> list[tuple[str, Headline]]:
    """Reported changes the record has still not caught up with.

    A headline is given ``grace_days`` for the record to follow; after
    that, if the club's most recent appointment is no later than a week
    before the story, the record is probably behind.
    """
    stale = []
    for club, headline in changes:
        if (today - headline.published).days < grace_days:
            continue
        appointed = latest_appointment.get(club)
        if appointed is None or appointed < headline.published - dt.timedelta(days=7):
            stale.append((club, headline))
    return stale
