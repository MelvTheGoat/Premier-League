#!/usr/bin/env python3
"""The job to run after every gameweek.

Fetch the latest results, rebuild the feature table, retrain both models
on everything now known, and predict the next gameweek.

    # Normal weekly run: ingest, retrain, predict the next gameweek
    python scripts/run_pipeline.py

    # First run of a season: also replay the gameweeks already played,
    # each predicted by a model that saw only what came before it
    python scripts/run_pipeline.py --backfill

    # Re-predict one gameweek
    python scripts/run_pipeline.py --matchday 7

Cron it for a few hours after the last fixture of a typical gameweek:

    0 6 * * TUE  cd /path/to/repo && .venv/bin/python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.pipeline import run as pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=config.CURRENT_SEASON)
    parser.add_argument("--matchday", type=int, help="Predict this gameweek only.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Also predict completed gameweeks that have no prediction yet.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the source refresh and use the local checkouts as they are.",
    )
    args = parser.parse_args()

    results = pipeline.run(
        season=args.season,
        matchday=args.matchday,
        refresh_source=not args.offline,
        backfill=args.backfill,
    )
    if not results:
        print("Nothing to predict: every gameweek in the season already has predictions.")
        return 0

    latest = results[-1]
    print(f"\nPredictions for {latest.season} gameweek {latest.matchday}:")
    for row in latest.predictions.to_dict("records"):
        print(
            f"  {row['home_team']:>24} v {row['away_team']:<24} "
            f"{row['predicted_outcome']}  "
            f"({row['p_home']:.2f}/{row['p_draw']:.2f}/{row['p_away']:.2f})  "
            f"{row['pred_home_goals']}-{row['pred_away_goals']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
