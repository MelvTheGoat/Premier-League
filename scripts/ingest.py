#!/usr/bin/env python3
"""Fetch and store source data without training anything.

Useful for checking a data source in isolation, or for priming the
database before the first pipeline run.

    python scripts/ingest.py
    python scripts/ingest.py --offline --first-season 2015-16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.data import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", default=config.FIRST_TRAINING_SEASON)
    parser.add_argument("--last-season", default=config.CURRENT_SEASON)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    report = ingest.run(
        refresh_source=not args.offline,
        first_season=args.first_season,
        last_season=args.last_season,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
