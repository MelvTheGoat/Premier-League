#!/usr/bin/env python3
"""Build the small, read-only database the website serves from.

Run this after every pipeline run; the site reads its output, not the
pipeline's working database.

    python scripts/export_web_db.py
    python scripts/export_web_db.py --season 2026-27   # current season only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.web.export import export  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.DB_PATH)
    parser.add_argument("--out", type=Path, default=config.WEB_DB_PATH)
    parser.add_argument(
        "--season",
        help="Export one season only. Omit to include every season on record.",
    )
    args = parser.parse_args()

    if not Path(args.source).is_file():
        print(f"No database at {args.source}; run the pipeline first.", file=sys.stderr)
        return 1

    counts = export(Path(args.source), Path(args.out), args.season)
    size = Path(args.out).stat().st_size / 1e6
    print(f"Wrote {args.out} ({size:.1f} MB)")
    for table, count in counts.items():
        print(f"  {table:22s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
