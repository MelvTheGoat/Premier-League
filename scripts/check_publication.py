#!/usr/bin/env python3
"""Assert that the database being served actually forecasts every gameweek.

A scheduled job that quietly stops doing its job looks exactly like a
scheduled job with nothing to do: both print "nothing published" and
both go green. That happened - a branch rename left the job publishing
to a branch the site was not deployed from, and eleven runs in a row
reported success while the site sat on a fortnight-old gameweek.

So the run does not get to end on its own say-so. This checks the
published database against the fixture list and fails if a gameweek that
has kicked off has no prediction of record, or if the gameweek about to
be played has not been forecast. A failing scheduled workflow emails the
repository owner, which is the point: silence should mean healthy, and
anything else should be noisy.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.web import queries  # noqa: E402


def _report_gameweek(matchday: int) -> None:
    """Hand the current gameweek to the workflow step that checks the site."""
    print(f"gameweek={matchday}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"gameweek={matchday}\n")


def _predicted_gameweeks(conn: sqlite3.Connection, season: str) -> set[int]:
    rows = conn.execute(
        "SELECT DISTINCT matchday FROM current_predictions WHERE season = ?",
        (season,),
    )
    return {int(row[0]) for row in rows}


def _started_gameweeks(conn: sqlite3.Connection, season: str) -> set[int]:
    """Gameweeks with at least one result in, so they cannot still be forecast."""
    rows = conn.execute(
        """
        SELECT DISTINCT matchday FROM matches
        WHERE season = ? AND competition = ? AND status = 'played'
          AND matchday IS NOT NULL
        """,
        (season, config.TARGET_COMPETITION),
    )
    return {int(row[0]) for row in rows}


def _next_unplayed_gameweek(conn: sqlite3.Connection, season: str) -> int | None:
    row = conn.execute(
        """
        SELECT MIN(matchday) FROM matches
        WHERE season = ? AND competition = ? AND status != 'played'
          AND matchday IS NOT NULL
        """,
        (season, config.TARGET_COMPETITION),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def check(db_path: Path, season: str) -> tuple[list[str], int | None]:
    """The reasons the published database is unfit, and the gameweek it leads with."""
    if not db_path.is_file():
        return [f"no serving database at {db_path}"], None

    problems: list[str] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        predicted = _predicted_gameweeks(conn, season)
        started = _started_gameweeks(conn, season)
        upcoming = _next_unplayed_gameweek(conn, season)

        missing = sorted(started - predicted)
        if missing:
            problems.append(
                "gameweeks that have been played carry no prediction of record: "
                + ", ".join(str(gameweek) for gameweek in missing)
            )

        if upcoming is not None and upcoming not in predicted:
            problems.append(
                f"gameweek {upcoming} is the next to be played and has not been forecast"
            )

        leading = queries.current_matchday(conn, season)

        print(
            f"{db_path.name}: {len(started)} gameweeks played, "
            f"{len(predicted)} forecast, next up {upcoming}, "
            f"site leads with gameweek {leading}"
        )
    finally:
        conn.close()

    return problems, leading


def main() -> int:
    db_path = Path(config.WEB_DB_PATH)
    problems, leading = check(db_path, config.CURRENT_SEASON)
    if leading is not None:
        _report_gameweek(leading)
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("Published database is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
