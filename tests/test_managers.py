"""Managerial spells from Wikidata, the merge with the hand-kept file, and the news tripwire."""

from __future__ import annotations

import datetime as dt
import shutil

from plpredict import config, db
from plpredict.data import ingest, teams
from plpredict.data.sources import manager_news, wikidata_managers
from plpredict.data.sources.wikidata_managers import Spell

# --- tidying what Wikidata returns --------------------------------------


def test_a_spell_recorded_from_both_directions_is_merged():
    spells = wikidata_managers.tidy(
        [
            Spell("Arsenal", "Mikel Arteta", "2019-12-20", None),
            Spell("Arsenal", "Mikel Arteta", "2019-12-22", None),
        ]
    )
    assert spells == [Spell("Arsenal", "Mikel Arteta", "2019-12-20", None)]


def test_an_open_spell_is_closed_when_the_next_manager_starts():
    spells = wikidata_managers.tidy(
        [
            Spell("Everton", "Sean Dyche", "2023-01-30", None),
            Spell("Everton", "David Moyes", "2025-01-11", None),
        ]
    )
    assert spells[0].end_date == "2025-01-10"
    assert spells[1].end_date is None


def test_a_year_only_record_beside_an_exact_one_is_dropped():
    """'2022' reads as 1 January, which would invent a spell six months early."""
    kept = wikidata_managers.drop_imprecise(
        [
            (Spell("Manchester United", "Erik ten Hag", "2022-01-01", "2024-01-01"), True),
            (Spell("Manchester United", "Erik ten Hag", "2022-07-01", "2024-10-28"), False),
            (Spell("Fulham", "Someone", "2012-01-01", None), True),  # nothing better known
        ]
    )
    assert Spell("Manchester United", "Erik ten Hag", "2022-07-01", "2024-10-28") in kept
    assert all(s.start_date != "2022-01-01" for s in kept)
    assert Spell("Fulham", "Someone", "2012-01-01", None) in kept


def test_coverage_counts_dates_inside_a_spell():
    spells = [Spell("Leeds United", "A", "2020-01-01", "2020-06-30")]
    dates = [("Leeds United", "2020-03-01"), ("Leeds United", "2020-08-01"), ("Fulham", "2020-03-01")]
    assert wikidata_managers.coverage(spells, dates) == 1 / 3


# --- the merge -----------------------------------------------------------


def test_the_hand_kept_file_only_fills_gaps(tmp_path, monkeypatch, request):
    wikidata_file = tmp_path / "managers_wikidata.csv"
    wikidata_managers.write(
        [
            Spell("Coventry City", "Mark Robins", "2017-03-06", "2024-11-07"),
            Spell("Coventry City", "Frank Lampard", "2024-11-28", None),
        ],
        wikidata_file,
    )
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "managers.csv").write_text(
        "# comment\nteam,manager,start_date,end_date,source\n"
        "Coventry City,Frank Lampard,2023-11-29,,manual\n"  # wrong year: covered, ignored
        "Burnley,Scott Parker,2024-07-01,,manual\n"  # nothing in Wikidata: kept
    )
    # Club names are resolved through the alias file in the same folder,
    # and the resolved table is cached: give it the real one, and leave no
    # cache behind built from this temporary folder.
    shutil.copy(config.MANUAL_DIR / "team_aliases.csv", manual_dir / "team_aliases.csv")
    monkeypatch.setattr(config, "WIKIDATA_MANAGERS_FILE", wikidata_file)
    monkeypatch.setattr(config, "MANUAL_DIR", manual_dir)
    teams.reset_alias_cache()
    request.addfinalizer(teams.reset_alias_cache)

    with db.connect(tmp_path / "t.db") as conn:
        conn.execute(
            "INSERT INTO managers (team, manager, start_date, source) VALUES ('Stale', 'Gone', '2000-01-01', 'old')"
        )
        ingest.ingest_managers(conn)
        rows = conn.execute("SELECT team, manager, start_date, source FROM managers ORDER BY team, start_date").fetchall()

    assert [tuple(r) for r in rows] == [
        ("Burnley", "Scott Parker", "2024-07-01", "manual"),
        ("Coventry City", "Mark Robins", "2017-03-06", "wikidata"),
        ("Coventry City", "Frank Lampard", "2024-11-28", "wikidata"),
    ]


# --- the news tripwire ---------------------------------------------------

CLUBS = ["Liverpool", "Manchester City", "Tottenham Hotspur", "Chelsea"]
DAY = dt.date(2026, 10, 1)


def _headline(title: str, days_ago: int = 5, summary: str = "") -> manager_news.Headline:
    return manager_news.Headline(title, summary, DAY - dt.timedelta(days=days_ago))


def test_a_sacking_is_picked_up_and_attributed():
    found = manager_news.manager_changes([_headline("Tottenham sack head coach after derby defeat")], CLUBS)
    assert [club for club, _ in found] == ["Tottenham Hotspur"]


def test_headline_nicknames_are_recognised():
    found = manager_news.manager_changes([_headline("Spurs sack boss after five defeats")], CLUBS)
    assert [club for club, _ in found] == ["Tottenham Hotspur"]


def test_stories_about_other_people_are_ignored():
    headlines = [
        _headline("Ward appointed Liverpool's sporting director"),
        _headline("Man City appoint new academy manager"),
        _headline("Chelsea boss named manager of the month"),
        _headline("Liverpool and Chelsea managers part ways with tradition"),  # two clubs
    ]
    assert manager_news.manager_changes(headlines, CLUBS) == []


def test_only_changes_the_record_has_not_caught_up_with_are_flagged():
    changes = [
        ("Chelsea", _headline("Chelsea sack manager", days_ago=10)),
        ("Liverpool", _headline("Liverpool appoint new head coach", days_ago=6)),
        ("Tottenham Hotspur", _headline("Tottenham sack head coach", days_ago=1)),  # within grace
    ]
    latest = {"Chelsea": dt.date(2026, 7, 1), "Liverpool": DAY - dt.timedelta(days=5)}
    stale = manager_news.unconfirmed(changes, latest, today=DAY)
    assert [club for club, _ in stale] == ["Chelsea"]


def test_the_feed_is_parsed():
    raw = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Spurs sack boss</title><description>Statement</description>
            <pubDate>Tue, 29 Sep 2026 10:00:00 GMT</pubDate></item>
      <item><title>No date</title></item>
    </channel></rss>"""
    [headline] = manager_news.parse_feed(raw)
    assert headline.title == "Spurs sack boss" and headline.published == dt.date(2026, 9, 29)
