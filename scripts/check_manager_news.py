#!/usr/bin/env python3
"""Flag managerial changes in the news that the record has not caught up with.

Advisory only: it never fails a run and never writes data. Findings are
printed and, under GitHub Actions, added to the run summary. See
``plpredict/data/sources/manager_news.py``.

    python scripts/check_manager_news.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config, db  # noqa: E402
from plpredict.data.sources import manager_news  # noqa: E402


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path and lines:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def main() -> int:
    try:
        headlines = manager_news.parse_feed(manager_news.fetch())
    except Exception as error:
        print(f"warning: news feed unavailable ({error})")
        return 0

    with db.connect() as conn:
        clubs = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT home_team FROM matches WHERE season = ? AND competition = ?",
                (config.CURRENT_SEASON, config.TARGET_COMPETITION),
            )
        ]
        latest = {
            row[0]: dt.date.fromisoformat(row[1])
            for row in conn.execute("SELECT team, MAX(start_date) FROM managers GROUP BY team")
        }

    changes = manager_news.manager_changes(headlines, clubs)
    stale = manager_news.unconfirmed(changes, latest, dt.date.today())

    print(f"{len(headlines)} headlines, {len(changes)} about a managerial change, {len(stale)} not yet in the record")
    lines = []
    for club, headline in stale:
        line = f"- {club}: \"{headline.title}\" ({headline.published}) - not yet reflected in Wikidata"
        print(line)
        lines.append(line)
    if lines:
        _summary(["", "**Possible managerial changes not yet in the record:**", *lines])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
