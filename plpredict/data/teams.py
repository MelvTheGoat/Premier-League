"""Canonical club names.

The archive spells the same club several ways depending on the season
("Bournemouth", "AFC Bournemouth"), so every name entering the database
passes through here first. Without it a club's history silently splits
in two and its form features reset mid-archive.

Normalisation is rule-based (drop the club-type suffix/prefix, fold
punctuation) with an explicit override table for the cases rules cannot
reach. Overrides live in ``data/manual/team_aliases.csv`` so a new
spelling can be fixed without a code change.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from plpredict import config

# Club-type words that carry no identity and appear inconsistently.
_AFFIXES = ("FC", "AFC", "A.F.C.", "F.C.")

_PUNCTUATION_RE = re.compile(r"[.'`’]")
_WHITESPACE_RE = re.compile(r"\s+")

# Short, display-friendly names for the twenty clubs a page shows most.
# Purely cosmetic: the canonical name remains the key everywhere else.
DISPLAY_NAMES = {
    "Brighton & Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Sheffield Wednesday": "Sheffield Wed",
    "Queens Park Rangers": "QPR",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Stoke City": "Stoke",
    "Birmingham City": "Birmingham",
    "Huddersfield Town": "Huddersfield",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Charlton Athletic": "Charlton",
    "Wigan Athletic": "Wigan",
    "Derby County": "Derby",
    "Middlesbrough": "Middlesbrough",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
}


def _strip_affixes(name: str) -> str:
    tokens = name.split()
    while tokens and tokens[0].upper().rstrip(".") in {a.upper().rstrip(".") for a in _AFFIXES}:
        tokens = tokens[1:]
    while tokens and tokens[-1].upper().rstrip(".") in {a.upper().rstrip(".") for a in _AFFIXES}:
        tokens = tokens[:-1]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def _alias_table() -> dict[str, str]:
    """Load manual overrides keyed by their normalised form."""
    path = config.MANUAL_DIR / "team_aliases.csv"
    table: dict[str, str] = {}
    if not Path(path).is_file():
        return table
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            alias = (row.get("alias") or "").strip()
            canonical = (row.get("canonical") or "").strip()
            if alias and canonical:
                table[_fold(alias)] = canonical
    return table


def _fold(name: str) -> str:
    """Aggressive form used only for lookups, never for display."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCTUATION_RE.sub("", text)
    text = _strip_affixes(text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def canonical_name(name: str) -> str:
    """Map any spelling of a club to the one name used everywhere."""
    raw = _WHITESPACE_RE.sub(" ", (name or "").strip())
    if not raw:
        return ""
    key = _fold(raw)
    override = _alias_table().get(key)
    if override:
        return override
    # Title-cased rebuild of the folded key preserves "&" and hyphens
    # while dropping the affixes.
    stripped = _strip_affixes(_PUNCTUATION_RE.sub("", raw))
    return _WHITESPACE_RE.sub(" ", stripped).strip() or raw


def display_name(canonical: str) -> str:
    return DISPLAY_NAMES.get(canonical, canonical)


def reset_alias_cache() -> None:
    _alias_table.cache_clear()
