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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.pipeline import run as pipeline  # noqa: E402


def _report_changed(changed: bool) -> None:
    """Tell a CI job whether anything was produced.

    Written to GITHUB_OUTPUT when running under GitHub Actions so the
    workflow can decide whether to commit, and echoed either way so the
    same signal is visible in a terminal.
    """
    value = "true" if changed else "false"
    print(f"changed={value}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"changed={value}\n")


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
    parser.add_argument(
        "--skip-if-unchanged",
        action="store_true",
        help=(
            "Stop after ingest if no new results have arrived since the "
            "database the site is serving. Intended for a scheduled job, "
            "which runs on a calendar while fixtures do not."
        ),
    )
    args = parser.parse_args()

    results = pipeline.run(
        season=args.season,
        matchday=args.matchday,
        refresh_source=not args.offline,
        backfill=args.backfill,
        skip_if_unchanged=args.skip_if_unchanged,
    )
    _report_changed(bool(results))
    if not results:
        print("Nothing new to publish.")
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
