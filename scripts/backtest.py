#!/usr/bin/env python3
"""Walk-forward backtest: how the model would have done, gameweek by gameweek.

    python scripts/backtest.py --seasons 2023-24 2024-25 2025-26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config, db  # noqa: E402
from plpredict.features.build import build_features  # noqa: E402
from plpredict.models import evaluate  # noqa: E402
from plpredict.pipeline.run import load_match_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument(
        "--from-matchday",
        type=int,
        default=1,
        help="Skip the opening gameweeks, where every model is guessing.",
    )
    parser.add_argument("--no-scoreline", action="store_true")
    parser.add_argument("--json", type=Path, help="Write the full report here.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    features = build_features()
    with db.connect() as conn:
        matches = load_match_frame(conn)

    report = evaluate.walk_forward(
        features,
        matches,
        args.seasons,
        include_scoreline=not args.no_scoreline,
        min_matchday=args.from_matchday,
        verbose=args.verbose,
    )

    print(report.summary())
    print("\nPer season:")
    for season, metrics in sorted(report.by_season.items()):
        print(
            f"  {season}: n={metrics['n']:4d}  accuracy {metrics['accuracy']:.1%}  "
            f"log loss {metrics['log_loss']:.4f}"
        )
    print("\nConfusion (rows = actual, columns = predicted):")
    print("        H     D     A")
    for actual, row in report.confusion.items():
        print(f"   {actual}  " + "".join(f"{row[name]:6d}" for name in ("H", "D", "A")))
    print("\nCalibration:")
    for row in report.calibration:
        print(
            f"  {row['bin_low']:.1f}-{row['bin_high']:.1f}  predicted {row['predicted']:.3f}  "
            f"observed {row['observed']:.3f}  (n={row['n']})"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.__dict__, indent=2, default=str))
        print(f"\nFull report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
