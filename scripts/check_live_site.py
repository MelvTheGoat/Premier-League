#!/usr/bin/env python3
"""Confirm the deployed site is serving what was just published.

Everything upstream of the deployment can be green while the site is
weeks out of date, because pushing a commit and serving it are two
different systems joined by a branch name. When the repository's default
branch changed, the scheduled job carried on pushing to it and going
green while the host went on deploying a branch that no longer received
anything. Eleven consecutive successful runs, a static site.

Nothing inside the repository can detect that, because from the inside
it looks perfect. So the job asks the site itself. Deployments are not
instant, so this polls for a few minutes before giving up.

    python scripts/check_live_site.py --gameweek 6
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict import config  # noqa: E402

# The landing page redirects to whichever gameweek is current and titles
# it in a heading. That heading is the one thing a reader would check.
_HEADING = re.compile(r"<h1>\s*Gameweek\s+(\d+)\s*</h1>", re.IGNORECASE)


def served_gameweek(url: str, timeout: float = 30.0) -> int | None:
    """The gameweek the live landing page is currently showing."""
    request = urllib.request.Request(url, headers={"User-Agent": "plpredict-check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    match = _HEADING.search(body)
    return int(match.group(1)) if match else None


def wait_for(url: str, expected: int, attempts: int, interval: float) -> tuple[bool, str]:
    """Poll until the site shows ``expected``, or the attempts run out."""
    last = "the site was never reached"
    for attempt in range(1, attempts + 1):
        try:
            gameweek = served_gameweek(url)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if gameweek == expected:
                return True, f"serving gameweek {gameweek}"
            last = (
                f"serving gameweek {gameweek}"
                if gameweek is not None
                else "no gameweek heading on the landing page"
            )

        print(f"  attempt {attempt}/{attempts}: {last}")
        if attempt < attempts:
            time.sleep(interval)

    return False, last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
        help="The gameweek the site should be showing.",
    )
    parser.add_argument("--url", default=config.SITE_URL)
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="How many times to look before failing (deploys take a minute or two).",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"
    print(f"Checking {url} for gameweek {args.gameweek}")

    ok, detail = wait_for(url, args.gameweek, args.attempts, args.interval)
    if ok:
        print(f"Live site is current: {detail}.")
        return 0

    print(
        f"FAIL: the site never showed gameweek {args.gameweek} ({detail}).\n"
        "The pipeline published, so the break is between the push and the\n"
        "deployment: check that the host is deploying the branch this job\n"
        "pushes to.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
