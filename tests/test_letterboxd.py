"""Tests for the export parser.

The fixtures reproduce the three traps found in a real export: diary files
keyed by entry URI rather than film URI, the three-block list format with CRLF
line endings, and a film whose watch record survives only in ``deleted/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lbrec.letterboxd import (
    coverage_report,
    film_status,
    load_export,
    normalize_title,
    read_list_csv,
)

RATINGS = """Date,Name,Year,Letterboxd URI,Rating
2025-04-04,12 Angry Men,1957,https://boxd.it/2auI,5
2025-04-04,Stalker,1979,https://boxd.it/2a4M,4.5
"""

WATCHED = """Date,Name,Year,Letterboxd URI
2025-03-22,12 Angry Men,1957,https://boxd.it/2auI
2025-03-22,Stalker,1979,https://boxd.it/2a4M
2025-03-22,Anora,2024,https://boxd.it/Egcw
"""

WATCHLIST = """Date,Name,Year,Letterboxd URI
2025-03-22,No Other Land,2024,https://boxd.it/KOTG
2025-03-22,Arco,2025,https://boxd.it/uh7C
2025-03-22,Anora,2024,https://boxd.it/Egcw
"""

LIKES = """Date,Name,Year,Letterboxd URI
2025-03-22,12 Angry Men,1957,https://boxd.it/2auI
"""

# Note the 6-char slugs: these are diary-entry URIs, not film URIs.
DIARY = """Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date
2026-05-19,Stalker,1979,https://boxd.it/epIkDx,4.5,Yes,,2026-03-21
"""

# A watch that survives only here: the film entry was deleted, so it never
# reaches watched.csv, while the watchlist row remains.
DELETED_DIARY = """Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date
2026-06-25,Arco,2025,https://boxd.it/eW77Vh,3.5,,,2026-06-24
"""

# Three blocks, CRLF endings, as Letterboxd actually writes them.
LIST = (
    "Letterboxd list export v7\r\n"
    "Date,Name,Tags,URL,Description\r\n"
    "2025-08-20,phase 2,,https://boxd.it/NHrKy,\r\n"
    "\r\n"
    "Position,Name,Year,URL,Description\r\n"
    "1,Ant-Man,2015,https://boxd.it/3vmW,\r\n"
    "2,Anora,2024,https://boxd.it/Egcw,\r\n"
)


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    files = {
        "ratings.csv": RATINGS,
        "watched.csv": WATCHED,
        "watchlist.csv": WATCHLIST,
        "diary.csv": DIARY,
        "likes/films.csv": LIKES,
        "deleted/diary.csv": DELETED_DIARY,
        "lists/phase-2.csv": LIST,
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    return tmp_path


def test_normalize_title_strips_accents_case_and_punctuation():
    assert normalize_title("Amélie!") == "amelie"
    assert normalize_title("Adaptation.") == "adaptation"
    assert normalize_title("  WALL·E  ") == "wall e"


def test_read_list_csv_parses_three_blocks(export_dir: Path):
    parsed = read_list_csv(export_dir / "lists/phase-2.csv")
    assert parsed is not None
    assert parsed.name == "phase 2"
    assert [entry["Name"] for entry in parsed.entries] == ["Ant-Man", "Anora"]


def test_read_list_csv_rejects_non_list_file(export_dir: Path):
    assert read_list_csv(export_dir / "ratings.csv") is None


def test_diary_joins_on_title_year_not_uri(export_dir: Path):
    """The diary URI is an entry URI; the row must still land on the film key."""
    interactions = load_export(export_dir).interactions
    diary = interactions[(interactions["kind"] == "diary") & (interactions["title"] == "Stalker")]
    assert len(diary) == 1
    assert diary["film_key"].item() == "2a4M"
    assert bool(diary["rewatch"].item()) is True
    assert "epIkDx" not in set(interactions["film_key"])


def test_missing_optional_files_are_not_an_error(tmp_path: Path):
    (tmp_path / "ratings.csv").write_text(RATINGS, encoding="utf-8", newline="")
    export = load_export(tmp_path)
    assert len(export.interactions) == 2
    assert set(export.interactions["kind"]) == {"rating"}


def test_deleted_entries_are_flagged_not_dropped(export_dir: Path):
    interactions = load_export(export_dir).interactions
    arco = interactions[(interactions["title"] == "Arco") & (interactions["kind"] == "diary")]
    assert len(arco) == 1
    assert bool(arco["deleted"].item()) is True

    dropped = load_export(export_dir, include_deleted=False).interactions
    assert dropped[dropped["title"] == "Arco"]["kind"].tolist() == ["watchlist"]


def test_seen_includes_films_missing_from_watched_csv(export_dir: Path):
    """Arco is diarised but absent from watched.csv, so it must still count as seen."""
    status = film_status(load_export(export_dir).interactions).set_index("film_key")
    assert bool(status.loc["uh7C", "seen"]) is True
    assert bool(status.loc["uh7C", "on_watchlist"]) is True
    assert bool(status.loc["uh7C", "watchlist_pending"]) is False


def test_watchlist_pending_excludes_seen_films(export_dir: Path):
    status = film_status(load_export(export_dir).interactions)
    pending = set(status.loc[status["watchlist_pending"], "title"])
    assert pending == {"No Other Land"}  # Anora is watched, Arco is diarised


def test_rating_prefers_ratings_csv_over_diary(export_dir: Path):
    status = film_status(load_export(export_dir).interactions).set_index("film_key")
    assert status.loc["2a4M", "rating"] == 4.5  # both sources agree
    assert status.loc["uh7C", "rating"] == 3.5  # diary fills a gap
    assert bool(status.loc["2auI", "liked"]) is True


def test_coverage_report_marks_optional_kinds(export_dir: Path):
    report = coverage_report(load_export(export_dir).interactions).set_index("kind")
    assert bool(report.loc["diary", "optional"]) is True
    assert bool(report.loc["rating", "optional"]) is False
    assert report.loc["rating", "rows"] == 2
