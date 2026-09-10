"""Name normalisation decides whether a club's history stays in one piece."""

from __future__ import annotations

import pytest

from plpredict.data.teams import canonical_name, display_name


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("Manchester United FC", "Manchester United"),
        ("Manchester United", "Manchester United"),
        ("Man United", "Manchester United"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Bournemouth", "Bournemouth"),
        ("Brighton & Hove Albion FC", "Brighton & Hove Albion"),
        ("Brighton", "Brighton & Hove Albion"),
        ("Tottenham", "Tottenham Hotspur"),
        ("Spurs", "Tottenham Hotspur"),
        ("Nott'm Forest", "Nottingham Forest"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("Sunderland AFC", "Sunderland"),
        ("Hull City AFC", "Hull City"),
        ("QPR", "Queens Park Rangers"),
    ],
)
def test_spellings_collapse_to_one_name(spelling, expected):
    assert canonical_name(spelling) == expected


def test_normalisation_is_idempotent():
    for name in ("Manchester United FC", "Man United", "AFC Bournemouth"):
        once = canonical_name(name)
        assert canonical_name(once) == once


def test_blank_names_survive():
    assert canonical_name("") == ""
    assert canonical_name(None) == ""


def test_display_name_falls_back_to_canonical():
    assert display_name("Manchester United") == "Man United"
    assert display_name("Arsenal") == "Arsenal"
