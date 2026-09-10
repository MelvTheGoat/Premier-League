#!/usr/bin/env python3
"""Build data/external/squad_ratings.csv from FIFA / EA FC rating dumps.

The rating dumps are large and are not vendored into this repository.
Point this script at whichever ones you have; every season it can cover
gets a row, and the feature layer carries the most recent rating forward
for seasons that are missing (recording how stale it is).

    python scripts/build_squad_ratings.py \
        --fifa-model-dir ../fifa_model/stats \
        --sofifa players_22.csv --sofifa fc26_players.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config, db  # noqa: E402
from plpredict.data.sources import fifa_ratings  # noqa: E402


def teams_by_season() -> dict[str, set[str]]:
    """English clubs per season, used to reject same-named foreign clubs."""
    mapping: dict[str, set[str]] = {}
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT season, home_team FROM matches "
            "WHERE competition IN ('premier_league', 'championship') "
            "UNION SELECT season, away_team FROM matches "
            "WHERE competition IN ('premier_league', 'championship')"
        )
        for season, team in rows:
            mapping.setdefault(season, set()).add(team)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fifa-model-dir",
        type=Path,
        help="Directory of player_stats_YYYY.csv files (lbenz730/fifa_model).",
    )
    parser.add_argument(
        "--sofifa",
        type=Path,
        action="append",
        default=[],
        help="A sofifa-style players CSV. Repeatable.",
    )
    parser.add_argument("--out", type=Path, default=config.EXTERNAL_DIR / "squad_ratings.csv")
    args = parser.parse_args()

    if not args.fifa_model_dir and not args.sofifa:
        parser.error("Give at least one of --fifa-model-dir or --sofifa")

    frame = fifa_ratings.build(
        fifa_model_dir=args.fifa_model_dir,
        sofifa_files=args.sofifa,
        teams_by_season=teams_by_season(),
    )
    if frame.empty:
        print("No squad ratings produced - check the source paths.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    seasons = sorted(frame["season"].unique())
    print(
        f"Wrote {len(frame)} club-season ratings to {args.out}\n"
        f"Seasons covered: {seasons[0]} - {seasons[-1]} ({len(seasons)} seasons)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
