"""The guards that stop a broken job from reporting success.

Both of these exist because of the same failure: for eleven consecutive
runs the scheduled job went green while the deployed site sat on a
fortnight-old gameweek. Nothing upstream was wrong, so nothing upstream
could have noticed — the push and the deployment had come apart at the
branch name. These are the two places that would now catch it.
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_live_site import served_gameweek, wait_for  # noqa: E402
from check_publication import check  # noqa: E402

from plpredict import db  # noqa: E402
from plpredict.features.build import FeatureBuilder  # noqa: E402
from plpredict.pipeline.run import (  # noqa: E402
    last_completed_gameweek,
    load_match_frame,
    next_unplayed_gameweek,
    train_and_predict,
)
from plpredict.web.export import export  # noqa: E402

SEASON = "2026-27"


@pytest.fixture(scope="module")
def served_db(seeded_db, tmp_path_factory):
    """A serving database built the way the real export builds one."""
    with db.connect(seeded_db) as conn:
        features = FeatureBuilder(conn).build()
        matches = load_match_frame(conn)
        completed = last_completed_gameweek(conn, SEASON)
        upcoming = next_unplayed_gameweek(conn, SEASON)
        for matchday in list(range(1, completed + 1)) + [upcoming]:
            train_and_predict(
                conn, features, matches, SEASON, matchday, min_feature_coverage=20
            )

    destination = tmp_path_factory.mktemp("web") / "plpredict-web.db"
    export(source=seeded_db, destination=destination, season=SEASON)
    return destination


def test_a_fully_forecast_season_passes(served_db):
    problems, leading = check(served_db, SEASON)
    assert problems == []
    assert leading is not None


def test_a_played_gameweek_with_no_prediction_is_caught(served_db, tmp_path):
    """The failure mode where the job quietly stops forecasting."""
    copy = tmp_path / "gap.db"
    copy.write_bytes(served_db.read_bytes())

    with db.connect(copy) as conn:
        dropped = conn.execute(
            "SELECT MIN(matchday) FROM current_predictions WHERE season = ?",
            (SEASON,),
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM current_predictions WHERE season = ? AND matchday = ?",
            (SEASON, dropped),
        )
        conn.commit()

    problems, _ = check(copy, SEASON)
    assert any(str(dropped) in problem for problem in problems)


def test_a_missing_database_is_caught(tmp_path):
    problems, leading = check(tmp_path / "absent.db", SEASON)
    assert problems and leading is None


class _Page(BaseHTTPRequestHandler):
    gameweek = 1

    def do_GET(self):  # noqa: N802 - the name is the server's, not ours
        body = f"<h1>Gameweek {self.gameweek}</h1>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def site():
    """A stand-in for the deployed site, serving one gameweek heading."""
    server = HTTPServer(("127.0.0.1", 0), _Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()


def test_the_live_page_is_read(site):
    _Page.gameweek = 7
    assert served_gameweek(site) == 7


def test_a_current_site_passes(site):
    _Page.gameweek = 7
    ok, detail = wait_for(site, expected=7, attempts=1, interval=0)
    assert ok, detail


def test_a_stale_site_fails(site):
    """The one the eleven green runs should have failed on."""
    _Page.gameweek = 5
    ok, detail = wait_for(site, expected=6, attempts=2, interval=0)
    assert not ok
    assert "gameweek 5" in detail


def test_an_unreachable_site_fails():
    # Port 1 on loopback refuses immediately, so this does not hang.
    ok, detail = wait_for("http://127.0.0.1:1/", expected=6, attempts=1, interval=0)
    assert not ok
    assert detail
