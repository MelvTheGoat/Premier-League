#!/usr/bin/env python3
"""Refresh every club's managerial history from Wikidata.

Writes ``data/external/managers_wikidata.csv``, which the ingest reads
ahead of the hand-kept ``data/manual/managers.csv``. Run daily by the
scheduled job, so an appointment reaches the features within a day of
being recorded, and committed when it changes:

    python scripts/sync_managers.py

A failed fetch keeps the existing file and reports it. Unlike the
availability log, nothing is lost by a missed day: the next successful
sync brings everything up to date.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config, db  # noqa: E402
from plpredict.data.sources import wikidata_managers as wikidata  # noqa: E402

# A refresh returning fewer spells than this share of the existing record
# is treated as a failed fetch.
MIN_SHARE_KEPT = 0.5


def _report(changed: bool) -> None:
    value = "true" if changed else "false"
    print(f"managers_changed={value}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"managers_changed={value}\n")


def main() -> int:
    output = Path(config.WIKIDATA_MANAGERS_FILE)
    club_ids = wikidata.read_club_ids(config.MANUAL_DIR / "team_wikidata.csv")

    # The club list comes from the pinned mapping, plus any club in the
    # fixture list not yet pinned. It must not depend on the database
    # alone: on a fresh runner this runs before the database is built.
    with db.connect() as conn:
        team_dates = [
            (row[0], row[1])
            for row in conn.execute(
                """
                SELECT home_team, match_date FROM matches
                WHERE competition = ? AND season >= ? AND match_date IS NOT NULL
                UNION ALL
                SELECT away_team, match_date FROM matches
                WHERE competition = ? AND season >= ? AND match_date IS NOT NULL
                """,
                (config.TARGET_COMPETITION, config.FIRST_TRAINING_SEASON) * 2,
            )
        ]
    clubs = sorted(set(club_ids) | {team for team, _ in team_dates})
    existing = wikidata.read(output)

    try:
        unpinned = [team for team in clubs if team not in club_ids]
        if unpinned:
            found = wikidata.resolve_club_ids(unpinned)
            print(f"Resolved by name (pin them in team_wikidata.csv): {found}")
            club_ids.update(found)
            missing = [team for team in unpinned if team not in found]
            if missing:
                print(f"warning: no Wikidata item found for {missing}")
        spells = wikidata.fetch_spells({t: q for t, q in club_ids.items() if t in clubs})
    except Exception as error:
        print(f"warning: Wikidata unavailable, keeping the existing file ({error})")
        _report(False)
        return 0

    # A partial answer is far more likely to be a failed query than a mass
    # sacking. Keep the record rather than overwrite it with less of it.
    if len(spells) < MIN_SHARE_KEPT * len(existing) or not spells:
        print(
            f"warning: Wikidata returned {len(spells)} spells against {len(existing)} "
            "on record; keeping the existing file"
        )
        _report(False)
        return 0

    before = output.read_bytes() if output.is_file() else b""
    wikidata.write(spells, output)
    changed = output.read_bytes() != before

    covered = (
        f"; {wikidata.coverage(spells, team_dates):.1%} of league matches since "
        f"{config.FIRST_TRAINING_SEASON} fall inside a known spell"
        if team_dates
        else ""
    )
    print(
        f"{len(spells)} spells for {len({s.team for s in spells})} clubs{covered}"
        + ("" if changed else "; unchanged")
    )
    _report(changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
