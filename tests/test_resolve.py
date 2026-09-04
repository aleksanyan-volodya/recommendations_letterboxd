"""Tests for TMDb ID resolution.

The matcher is deliberately conservative: a wrong ID silently attaches another
film's synopsis and popularity to a rating, which is far worse than a gap a
human fills in. These tests pin that bias toward refusing to guess.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from lbrec.resolve import (
    EXACT,
    HIGH,
    MEDIA_MOVIE,
    MEDIA_TV,
    MEDIUM,
    OVERRIDE,
    UNRESOLVED,
    Override,
    append_overrides,
    build_review_table,
    duplicate_ids,
    load_overrides,
    read_reviewed,
    resolve_film,
    resolve_films,
    score_candidates,
    sequel_marker,
    title_score,
)
from lbrec.tmdb import TmdbClient
from tests.test_tmdb import RecordingTransport, make_settings


def candidate(tmdb_id, title, year, *, original_title=None, popularity=1.0):
    return {
        "id": tmdb_id,
        "title": title,
        "original_title": original_title or title,
        "release_date": f"{year}-01-01" if year else "",
        "popularity": popularity,
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_exact_title_and_year_wins():
    match = score_candidates("12 Angry Men", 1957, [candidate(389, "12 Angry Men", 1957)])
    assert (match.tmdb_id, match.confidence) == (389, EXACT)


def test_punctuation_and_accents_do_not_block_an_exact_match():
    match = score_candidates("Amelie", 2001, [candidate(194, "Amélie", 2001)])
    assert match.confidence == EXACT


def test_matches_against_the_original_title():
    """Letterboxd may export a title TMDb only carries as original_title."""
    match = score_candidates(
        "Sanma no aji",
        1962,
        [candidate(18148, "An Autumn Afternoon", 1962, original_title="Sanma no aji")],
    )
    assert match.confidence == EXACT


def test_year_off_by_one_is_tolerated():
    """Release-year disagreement between Letterboxd and TMDb is routine."""
    match = score_candidates("Stalker", 1979, [candidate(1398, "Stalker", 1980)])
    assert match.confidence == EXACT


def test_wrong_year_is_rejected_even_with_an_identical_title():
    match = score_candidates("Solaris", 1972, [candidate(852, "Solaris", 2002)])
    assert match.confidence == UNRESOLVED
    assert match.tmdb_id is None


def test_unrelated_title_is_rejected():
    match = score_candidates("Stalker", 1979, [candidate(1, "Predator", 1979)])
    assert match.confidence == UNRESOLVED


def test_leading_article_difference_still_matches():
    """ "The Fisher King" vs "Fisher King" scores only 84.6 on raw similarity."""
    match = score_candidates("The Fisher King", 1991, [candidate(2, "Fisher King", 1991)])
    assert (match.tmdb_id, match.confidence) == (2, EXACT)


def test_trailing_article_form_still_matches():
    """TMDb sometimes stores the sorted form, "Fisher King, The"."""
    match = score_candidates("The Fisher King", 1991, [candidate(2, "Fisher King, The", 1991)])
    assert (match.tmdb_id, match.confidence) == (2, EXACT)


def test_article_stripping_does_not_match_unrelated_titles():
    """Variants may only raise the score; they must not defeat the year or title guard."""
    assert score_candidates("The Thing", 1982, [candidate(1, "The Thing", 2011)]).confidence == (
        UNRESOLVED
    )
    assert score_candidates("A Woman", 1915, [candidate(1, "A Man", 1915)]).confidence == UNRESOLVED


def test_near_miss_title_lands_below_exact():
    match = score_candidates("Blade Runner", 1982, [candidate(2, "Blade Runners", 1982)])
    assert match.confidence in {HIGH, MEDIUM}
    assert match.tmdb_id == 2


def test_sequel_numbers_must_agree():
    """Caught in a live run: "Drunken Master II" matched "Drunken Master III" at 97.1."""
    match = score_candidates(
        "Drunken Master II", 1994, [candidate(66018, "Drunken Master III", 1994)]
    )
    assert match.confidence == UNRESOLVED


def test_arabic_sequel_numbers_must_agree():
    assert score_candidates(
        "Toy Story 3", 2010, [candidate(1, "Toy Story 2", 2010)]
    ).confidence == (UNRESOLVED)


def test_a_trailing_number_on_only_one_side_is_a_conflict():
    """ "Blade Runner" and "Blade Runner 2049" are different films."""
    match = score_candidates("Blade Runner", 1982, [candidate(1, "Blade Runner 2049", 1982)])
    assert match.confidence == UNRESOLVED


def test_matching_sequel_numbers_still_resolve():
    match = score_candidates(
        "Drunken Master II", 1994, [candidate(11045, "Drunken Master II", 1994)]
    )
    assert (match.tmdb_id, match.confidence) == (11045, EXACT)


def test_single_token_numeric_titles_are_not_read_as_sequels():
    """A film called "X" or "1917" must not be treated as carrying a sequel marker."""
    assert sequel_marker("x") is None
    assert sequel_marker("1917") is None
    assert score_candidates("1917", 2019, [candidate(530915, "1917", 2019)]).confidence == EXACT


def test_sequel_guard_checks_every_title_form():
    """The marker may agree with original_title even when the localised title differs."""
    match = score_candidates(
        "Rocky II", 1979, [candidate(1367, "Rocky, Part II", 1979, original_title="Rocky II")]
    )
    assert match.confidence == EXACT


def test_missing_year_on_either_side_is_not_accepted():
    """Without a year to corroborate, a title match alone is not enough."""
    assert score_candidates("Stalker", None, [candidate(1398, "Stalker", 1979)]).confidence == (
        UNRESOLVED
    )
    assert score_candidates("Stalker", 1979, [candidate(1398, "Stalker", None)]).confidence == (
        UNRESOLVED
    )


def test_exact_match_beats_a_more_popular_inexact_one():
    candidates = [
        candidate(1, "The Stalker", 1979, popularity=99.0),
        candidate(1398, "Stalker", 1979, popularity=0.1),
    ]
    assert score_candidates("Stalker", 1979, candidates).tmdb_id == 1398


def test_popularity_breaks_ties_between_equally_exact_matches():
    """Identity resolution only -- the better-known film is the likelier referent."""
    candidates = [
        candidate(1, "Nosferatu", 1922, popularity=5.0),
        candidate(2, "Nosferatu", 1922, popularity=50.0),
    ]
    assert score_candidates("Nosferatu", 1922, candidates).tmdb_id == 2


def test_title_score_is_symmetric_across_title_fields():
    assert title_score("Ran", candidate(1, "Ran", 1985)) == 100.0
    assert title_score("Ran", candidate(1, "Chaos", 1985, original_title="Ran")) == 100.0


def test_empty_candidate_list_is_unresolved():
    assert score_candidates("Whatever", 1999, []).confidence == UNRESOLVED


# --------------------------------------------------------------------------
# search flow
# --------------------------------------------------------------------------
def test_resolve_film_falls_back_to_an_unconstrained_search(tmp_path: Path):
    """TMDb's primary_release_year filter can hide the right film; retry without it."""
    pages = [
        {"results": []},  # year-constrained search finds nothing
        {"results": [candidate(1398, "Stalker", 1979)]},  # unconstrained finds it
    ]
    transport = RecordingTransport(lambda r: httpx.Response(200, json=pages.pop(0)))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        match = resolve_film(client, "Stalker", 1979)
    assert (match.tmdb_id, match.confidence) == (1398, EXACT)
    assert len(transport.requests) == 2


def test_resolve_film_skips_the_second_search_on_an_exact_hit(tmp_path: Path):
    payload = {"results": [candidate(389, "12 Angry Men", 1957)]}
    transport = RecordingTransport(lambda r: httpx.Response(200, json=payload))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        resolve_film(client, "12 Angry Men", 1957)
    assert len(transport.requests) == 1


def tv_result(tmdb_id, name, year, *, original_name=None, popularity=1.0):
    return {
        "id": tmdb_id,
        "name": name,
        "original_name": original_name or name,
        "first_air_date": f"{year}-01-01" if year else "",
        "popularity": popularity,
    }


def test_television_is_found_when_the_film_namespace_has_nothing(tmp_path: Path):
    """Letterboxd lists miniseries that /search/movie structurally cannot return."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/search/tv" in request.url.path:
            return httpx.Response(200, json={"results": [tv_result(87108, "Chernobyl", 2019)]})
        return httpx.Response(200, json={"results": []})

    transport = RecordingTransport(handler)
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        match = resolve_film(client, "Chernobyl", 2019)
    assert (match.tmdb_id, match.media_type, match.confidence) == (87108, MEDIA_TV, EXACT)


def test_a_film_match_is_never_displaced_by_a_like_named_series(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "/search/tv" in request.url.path:
            return httpx.Response(200, json={"results": [tv_result(999, "Fargo", 1996)]})
        return httpx.Response(200, json={"results": [candidate(275, "Fargo", 1996)]})

    transport = RecordingTransport(handler)
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        match = resolve_film(client, "Fargo", 1996)
    assert (match.tmdb_id, match.media_type) == (275, MEDIA_MOVIE)


def test_tv_search_is_skipped_entirely_when_disallowed(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"results": []}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        resolve_film(client, "Chernobyl", 2019, allow_tv=False)
    assert all("/search/tv" not in r.url.path for r in transport.requests)


def test_blank_title_never_calls_the_api(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        assert resolve_film(client, "  ", 1999).confidence == UNRESOLVED
    assert transport.requests == []


# --------------------------------------------------------------------------
# batch resolution and overrides
# --------------------------------------------------------------------------
def test_overrides_win_without_touching_the_api(tmp_path: Path):
    films = pd.DataFrame([{"film_key": "2auI", "title": "12 Angry Men", "year": 1957}])
    transport = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        film_map = resolve_films(client, films, {"2auI": Override(389)})
    assert transport.requests == []
    assert film_map.loc[0, "tmdb_id"] == 389
    assert film_map.loc[0, "confidence"] == OVERRIDE
    assert film_map.loc[0, "media_type"] == MEDIA_MOVIE


def test_override_with_no_id_records_a_deliberate_non_match(tmp_path: Path):
    films = pd.DataFrame([{"film_key": "xxxx", "title": "Lost Film", "year": 1920}])
    transport = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        film_map = resolve_films(client, films, {"xxxx": Override(None)})
    assert transport.requests == []
    assert pd.isna(film_map.loc[0, "tmdb_id"])
    assert film_map.loc[0, "confidence"] == UNRESOLVED


def test_overrides_round_trip(tmp_path: Path):
    path = tmp_path / "overrides.csv"
    append_overrides(path, [{"film_key": "2auI", "tmdb_id": "389", "title": "x", "note": ""}])
    append_overrides(
        path, [{"film_key": "abcd", "tmdb_id": "", "title": "y", "note": "no TMDb entry"}]
    )
    append_overrides(
        path, [{"film_key": "mkbG", "tmdb_id": "87108", "media_type": "tv", "title": "Chernobyl"}]
    )
    assert load_overrides(path) == {
        "2auI": Override(389, MEDIA_MOVIE),
        "abcd": Override(None, MEDIA_MOVIE),
        "mkbG": Override(87108, MEDIA_TV),
    }


def test_a_later_override_row_supersedes_an_earlier_one(tmp_path: Path):
    """Corrections are appended, never rewritten, so the last decision wins."""
    path = tmp_path / "overrides.csv"
    append_overrides(path, [{"film_key": "B4Le", "tmdb_id": "258216", "title": "Nymphomaniac"}])
    append_overrides(
        path,
        [{"film_key": "B4Le", "tmdb_id": "249397", "title": "Nymphomaniac", "note": "Vol. II"}],
    )
    assert load_overrides(path)["B4Le"] == Override(249397, MEDIA_MOVIE)
    assert "258216" in path.read_text()  # the earlier decision is still on record


def test_decided_non_matches_do_not_come_back_to_review(tmp_path: Path):
    """A recorded "not on TMDb" must stop being asked about every run."""
    film_map = pd.DataFrame(
        [
            _review_row(film_key="aaaa", confidence=UNRESOLVED, source=OVERRIDE),
            _review_row(film_key="bbbb", confidence=UNRESOLVED, source="auto"),
        ]
    )
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"results": []}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        review = build_review_table(client, film_map)
    assert review["film_key"].tolist() == ["bbbb"]


def test_duplicate_ids_are_reported(tmp_path: Path):
    film_map = pd.DataFrame(
        [
            _review_row(film_key="7DiG", tmdb_id=258216, confidence=EXACT),
            _review_row(film_key="B4Le", tmdb_id=258216, confidence=OVERRIDE),
            _review_row(film_key="cccc", tmdb_id=999, confidence=EXACT),
            _review_row(film_key="dddd", tmdb_id=pd.NA, confidence=UNRESOLVED),
        ]
    )
    assert duplicate_ids(film_map)["film_key"].tolist() == ["7DiG", "B4Le"]


def test_hand_resolved_tv_keeps_its_namespace(tmp_path: Path):
    """A series resolved by hand must not later be looked up as a film."""
    films = pd.DataFrame([{"film_key": "mkbG", "title": "Chernobyl", "year": 2019}])
    transport = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        film_map = resolve_films(client, films, {"mkbG": Override(87108, MEDIA_TV)})
    assert film_map.loc[0, "media_type"] == MEDIA_TV


def test_append_overrides_never_rewrites_existing_rows(tmp_path: Path):
    path = tmp_path / "overrides.csv"
    append_overrides(path, [{"film_key": "2auI", "tmdb_id": "389", "title": "x", "note": ""}])
    before = path.read_text()
    append_overrides(path, [{"film_key": "abcd", "tmdb_id": "1", "title": "y", "note": ""}])
    assert path.read_text().startswith(before)


def test_read_reviewed_ignores_untouched_rows(tmp_path: Path):
    path = tmp_path / "unresolved.csv"
    path.write_text(
        "film_key,title,year,letterboxd_url,tmdb_id,suggestion_1,note\n"
        "aaaa,Filled In,1999,,123,,\n"
        "bbbb,Untouched,1999,,,,\n"
        "cccc,Deliberate Skip,1999,,,,not on TMDb\n",
        encoding="utf-8",
    )
    rows = read_reviewed(path)
    assert [row["film_key"] for row in rows] == ["aaaa", "cccc"]


def _review_row(**overrides):
    row = {
        "film_key": "zzzz",
        "title": "Obscure",
        "year": 1974,
        "confidence": UNRESOLVED,
        "source": "auto",
        "tmdb_id": pd.NA,
        "media_type": pd.NA,
        "tmdb_title": "",
        "tmdb_year": pd.NA,
    }
    return {**row, **overrides}


def test_build_review_table_offers_suggestions(tmp_path: Path):
    film_map = pd.DataFrame([_review_row()])
    payload = {"results": [candidate(7, "Obscure Film", 1975)]}
    transport = RecordingTransport(lambda r: httpx.Response(200, json=payload))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        review = build_review_table(client, film_map)
    assert review.loc[0, "tmdb_id"] == ""  # left blank for the human
    assert review.loc[0, "auto_match"] == ""  # nothing was matched
    assert review.loc[0, "suggestion_1"] == "7 | Obscure Film (1975)"
    assert review.loc[0, "letterboxd_url"] == "https://boxd.it/zzzz"


def test_medium_confidence_goes_to_review_not_straight_through(tmp_path: Path):
    """Medium is usually right, but confirming a few rows beats one silent mismatch."""
    film_map = pd.DataFrame(
        [
            _review_row(film_key="aaaa", confidence=EXACT, tmdb_id=1),
            _review_row(film_key="bbbb", confidence=HIGH, tmdb_id=2),
            _review_row(
                film_key="cccc",
                confidence=MEDIUM,
                tmdb_id=3,
                media_type=MEDIA_MOVIE,
                tmdb_title="Close",
                tmdb_year=1976,
            ),
        ]
    )
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"results": []}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        review = build_review_table(client, film_map)
    assert review["film_key"].tolist() == ["cccc"]
    assert review.loc[0, "auto_match"] == "3 | Close (1976) [medium, movie]"
    assert review.loc[0, "tmdb_id"] == ""
