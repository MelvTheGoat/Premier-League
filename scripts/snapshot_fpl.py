#!/usr/bin/env python3
"""Record current player availability from the FPL API.

The API only ever reports the present, so this is run every day and
appends whatever has changed since the last observation to a committed
log. See ``plpredict/data/sources/fpl.py`` for why.

    python scripts/snapshot_fpl.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402
from plpredict.data.sources import fpl  # noqa: E402


def _report(recorded: int) -> None:
    """Tell a CI job how many rows were written, so it knows to commit."""
    print(f"recorded={recorded}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"recorded={recorded}\n")


def main() -> int:
    log_path = Path(config.FPL_AVAILABILITY_LOG)
    try:
        payload = fpl.fetch_bootstrap(config.FPL_BOOTSTRAP_URL)
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        _report(0)
        return 1

    now = dt.datetime.now(dt.timezone.utc)
    snapshot = fpl.parse_bootstrap(payload, config.CURRENT_SEASON, now)
    recorded = fpl.record(snapshot, log_path)

    statuses = Counter(player.status for player in snapshot)
    flagged = sum(count for status, count in statuses.items() if status != "a")
    print(
        f"{len(snapshot)} players observed, {flagged} not fully available "
        f"({', '.join(f'{s}={n}' for s, n in sorted(statuses.items()))}); "
        f"{recorded} changes appended to {log_path.name}"
    )
    _report(recorded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
