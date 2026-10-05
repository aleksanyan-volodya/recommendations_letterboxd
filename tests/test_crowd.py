"""Tests for the Letterboxd crowd dumps and the candidate pool built from them.

The pool decides what may be recommended at all, so every way of silently
losing a film, inventing a rater, or handing an obscure film's audience to a
famous one is pinned here.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from lbrec.crowd import (
    AMBIGUOUS,
    DEAD_LINK,
    LINKED,
    MATCHED,
    NO_RECORD,
    UNMATCHED,
    as_model_ratings,
    build_pool,
    crowd_item_bias,
    latest_snapshot,
    map_slugs,
    mapping_report,
    match_by_title_year,
    pool_by_year,
    pool_summary,
    ratings_by_film,
    read_freeth_films,
    read_freeth_ratings,
    read_samlearner_films,
    read_samlearner_ratings,
)
from lbrec.movielens import rated_tmdb_ids

SAMLEARNER_RATINGS = (
    "_id,movie_id,rating_val,user_id\na1,null,7,alice\na2,nan,8,alice\na3,feast-2014,10,Bob\n"
)
MOVIE_DATA_HEADER = "_id,movie_id,movie_title,tmdb_id,year_released,overview\n"
MOVIE_DATA = (
    MOVIE_DATA_HEADER + "m1,null,Null,111,2019,a film called null\n"
    "m2,feast-2014,Feast,222,2014,short\n"
    "m3,broken,Broken,2001\n"  # truncated mid-record, as in the real dump
    "m4,no-link,No Link,,2005,no tmdb id\n"
)
FREETH_RATINGS = (
    "user_name,film_id,rating\ncarol,null,0.5\ncarol,feast-2014,5.0\nbob,feast-2014,3.5\n"
)
FREETH_FILMS = "film_id,film_name,year,poster_url\nnull,Null,2019,x\nnew-film,New Film,2023,\n"


@pytest.fixture
def samlearner(tmp_path: Path) -> Path:
    path = tmp_path / "samlearner.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ratings_export.csv", SAMLEARNER_RATINGS)
        z.writestr("movie_data.csv", MOVIE_DATA)
    return path


@pytest.fixture
def freeth(tmp_path: Path) -> Path:
    path = tmp_path / "freeth.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ratings.csv", FREETH_RATINGS)
        z.writestr("films.csv", FREETH_FILMS)
    return path


def ratings(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["member", "slug", "rating"])
    frame["rating"] = frame["rating"].astype("int8")
    return frame


def films(rows: list[tuple[str, int | None, str, int]]) -> pd.DataFrame:
    """A dump's film table: slug, TMDb link (or None), title, year."""
    frame = pd.DataFrame(rows, columns=["slug", "tmdb_id", "title", "year"])
    frame["tmdb_id"] = pd.array(frame["tmdb_id"].tolist(), "Int64")
    frame["year"] = frame["year"].astype("Int64")
    return frame


def catalogue(rows: list[tuple[int, str, int]], **extra) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["tmdb_id", "title", "year"])
    frame["original_title"] = frame["title"]
    frame["year"] = frame["year"].astype("Int64")
    for column, values in extra.items():
        frame[column] = values
    return frame


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def test_slugs_that_look_like_missing_values_are_films(samlearner: Path, freeth: Path):
    """pandas' default NA parsing drops the films whose slugs are 'null' and 'nan'."""
    old = read_samlearner_ratings(samlearner)
    assert {"null", "nan"} <= set(old["slug"])
    assert old["slug"].notna().all()
    assert "null" in set(read_freeth_ratings(freeth)["slug"])
    assert "null" in set(read_samlearner_films(samlearner)["slug"])
    assert "null" in set(read_freeth_films(freeth)["slug"])


def test_member_names_are_case_folded(samlearner: Path):
    """Letterboxd usernames are case-insensitive; two scrapers need not agree on case."""
    assert "bob" in set(read_samlearner_ratings(samlearner)["member"])


def test_freeth_stars_are_doubled_onto_the_ten_point_scale(freeth: Path):
    frame = read_freeth_ratings(freeth)
    assert sorted(frame["rating"].tolist()) == [1, 7, 10]
    assert frame["rating"].dtype == "int8"


def test_ratings_off_the_scale_are_refused(tmp_path: Path):
    """A schema surprise must be loud, not quietly folded into the averages."""
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ratings_export.csv", "_id,movie_id,rating_val,user_id\na,x,11,u\n")
    with pytest.raises(ValueError, match="1-10 scale"):
        read_samlearner_ratings(path)


def test_truncated_film_rows_are_skipped_and_counted(samlearner: Path):
    films = read_samlearner_films(samlearner)
    assert "broken" not in set(films["slug"])
    assert films.attrs["skipped_rows"] == 1


def test_films_without_a_tmdb_link_are_kept_for_matching(samlearner: Path):
    """Dropping them here would make them unrecoverable later."""
    films = read_samlearner_films(samlearner).set_index("slug")
    assert pd.isna(films.loc["no-link", "tmdb_id"])
    assert films.loc["feast-2014", "tmdb_id"] == 222


# --------------------------------------------------------------------------
# merging dumps
# --------------------------------------------------------------------------
def test_a_newer_snapshot_replaces_the_member_wholesale():
    """A film absent from the newer scrape was un-rated; merging per film would revive it."""
    old = ratings([("alice", "a", 6), ("alice", "deleted-since", 2)])
    new = ratings([("alice", "a", 9)])
    merged = latest_snapshot([("old", old), ("new", new)])
    assert merged[["slug", "rating"]].values.tolist() == [["a", 9]]
    assert merged["dump"].tolist() == ["new"]


def test_members_only_in_the_older_dump_are_kept():
    old = ratings([("alice", "a", 6), ("dave", "b", 4)])
    new = ratings([("alice", "a", 9)])
    merged = latest_snapshot([("old", old), ("new", new)])
    assert set(merged["member"]) == {"alice", "dave"}
    assert merged.attrs["superseded"] == {"new": 0, "old": 1}


# --------------------------------------------------------------------------
# slugs to TMDb ids
# --------------------------------------------------------------------------
def test_letterboxd_own_link_beats_a_title_match():
    """Most title disagreements are translations; the dump's link is the identity."""
    dump = films([("cobra-gang", 5, "Cobra Gang", 1985)])
    cat = catalogue([(5, "Rafaga De Plomo", 1985), (6, "Cobra Gang", 1985)])
    mapped = map_slugs([dump], cat).set_index("slug")
    assert mapped.loc["cobra-gang", "tmdb_id"] == 5
    assert mapped.loc["cobra-gang", "how"] == LINKED


def test_a_link_from_an_older_dump_survives_a_newer_dump_without_one():
    old = films([("x", 7, "X", 2000)])
    new = films([("x", None, "X", 2000)])
    mapped = map_slugs([old, new], catalogue([(7, "X", 2000)]))
    assert mapped["tmdb_id"].tolist() == [7]


def test_a_dead_link_falls_back_to_title_and_year():
    """TMDb deleted the id and Letterboxd relinked the page; the old id points nowhere."""
    dump = films([("lovers-rock", 5, "Lovers Rock", 2020)])
    cat = catalogue([(9, "Lovers Rock", 2020)])
    mapped = map_slugs([dump], cat, live_ids={9}).set_index("slug")
    assert mapped.loc["lovers-rock", "tmdb_id"] == 9
    assert mapped.loc["lovers-rock", "how"] == MATCHED


def test_a_dead_link_with_no_match_says_so():
    dump = films([("black-mirror-nosedive", 5, "Black Mirror: Nosedive", 2016)])
    mapped = map_slugs([dump], catalogue([(9, "Other", 2016)]), live_ids={9})
    assert mapped["tmdb_id"].isna().all()
    assert mapped["how"].tolist() == [DEAD_LINK]


def test_a_live_link_outside_the_catalogue_is_not_rematched():
    """A video release keeps its id, so its audience cannot pass to a namesake."""
    dump = films([("twin-peaks", 5, "Twin Peaks", 1989)])
    cat = catalogue([(9, "Twin Peaks", 1989)])
    mapped = map_slugs([dump], cat, live_ids={5, 9})
    assert mapped["tmdb_id"].tolist() == [5]
    assert mapped["how"].tolist() == [LINKED]


def unlinked(title: str, year: int | None) -> pd.DataFrame:
    return pd.DataFrame({"slug": ["s"], "title": [title], "year": pd.array([year], "Int64")})


def test_unlinked_slug_matches_a_unique_title_and_year():
    out = match_by_title_year(unlinked("New Film!", 2023), catalogue([(9, "New Film", 2023)]))
    assert out["tmdb_id"].tolist() == [9]
    assert out["how"].tolist() == [MATCHED]


def test_titles_must_match_literally_not_up_to_an_article():
    """Small Axe's "Education" moved to TV; "The Education" (2020) is another film."""
    out = match_by_title_year(unlinked("Education", 2020), catalogue([(9, "The Education", 2020)]))
    assert out["how"].tolist() == [UNMATCHED]


def test_an_exact_year_beats_an_adjacent_one():
    cat = catalogue([(1, "Parasite", 2019), (2, "Parasite", 2018)])
    assert match_by_title_year(unlinked("Parasite", 2019), cat)["tmdb_id"].tolist() == [1]


def test_ambiguous_title_and_year_is_not_handed_to_the_famous_film():
    """A popularity tie-break would give an obscure film's raters to its namesake."""
    cat = catalogue([(1, "Mother", 2009), (2, "Mother", 2009)], popularity=[90.0, 0.1])
    out = match_by_title_year(unlinked("Mother", 2009), cat)
    assert out["tmdb_id"].isna().all()
    assert out["how"].tolist() == [AMBIGUOUS]


def test_a_slug_without_a_year_is_never_matched():
    out = match_by_title_year(unlinked("Mother", None), catalogue([(1, "Mother", 2009)]))
    assert out["how"].tolist() == [UNMATCHED]


def test_a_different_sequel_number_is_a_different_film():
    out = match_by_title_year(
        unlinked("Drunken Master II", 1994), catalogue([(1, "Drunken Master", 1994)])
    )
    assert out["how"].tolist() == [UNMATCHED]


def test_original_titles_are_matched_too():
    cat = catalogue([(1, "Years Like Flowers", 2001)])
    cat["title"] = "Age of Bloom"
    out = match_by_title_year(unlinked("Years Like Flowers", 2001), cat)
    assert out["tmdb_id"].tolist() == [1]


# --------------------------------------------------------------------------
# ratings keyed by film, and the pool
# --------------------------------------------------------------------------
def keyed(rows: list[tuple[str, str, int]], slug_map: dict[str, int]) -> pd.DataFrame:
    merged = latest_snapshot([("d", ratings(rows))])
    table = pd.DataFrame(
        {"slug": list(slug_map), "tmdb_id": pd.array(list(slug_map.values()), "Int64")}
    )
    return ratings_by_film(merged, table)


def test_two_slugs_for_one_film_make_one_rater():
    """Renamed pages keep old slugs in older scrapes; one person is still one rater."""
    out = keyed(
        [("alice", "magnetic-rose", 6), ("alice", "magnetic-rose-1995", 9)],
        {"magnetic-rose": 30, "magnetic-rose-1995": 30},
    )
    assert out[["member", "tmdb_id"]].values.tolist() == [["alice", 30]]
    assert out["rating"].tolist() == [8]  # 7.5 rounds to even
    assert out.attrs["merged_duplicates"] == 2


def test_unmapped_slugs_carry_no_rating_forward():
    out = keyed([("alice", "mapped", 6), ("alice", "lost", 9)], {"mapped": 1})
    assert out["tmdb_id"].tolist() == [1]


def test_mapping_report_counts_what_unmapped_slugs_would_have_added_to_the_pool():
    """Rated slugs in no film table must show up as lost, not vanish."""
    merged = latest_snapshot(
        [("d", ratings([("a", "linked", 5), ("b", "linked", 5), ("a", "orphan", 5)]))]
    )
    slug_map = pd.DataFrame({"slug": ["linked"], "how": [LINKED]})
    report = mapping_report(merged, slug_map, min_raters=2).set_index("how")
    assert report.loc[LINKED, "pool_sized"] == 1
    assert report.loc[NO_RECORD, "slugs"] == 1
    assert report.loc[NO_RECORD, "pool_sized"] == 0


def test_model_ratings_use_integer_users_tmdb_items_and_the_five_star_scale():
    """A member's export is on 0.5-5; training on 1-10 would double every bias."""
    crowd = pd.DataFrame(
        {"member": ["ann", "bo", "ann"], "tmdb_id": [7, 7, 8], "rating": [10, 1, 7]}
    )
    out = as_model_ratings(crowd)
    assert out["rating"].tolist() == [5.0, 0.5, 3.5]
    assert out["movieId"].tolist() == [7, 7, 8]
    assert out["userId"].tolist() == [0, 1, 0]
    assert out["userId"].dtype == "int32"


def test_crowd_bias_is_on_the_models_scale_and_shrunk():
    """Stored ratings are 1-10; the models work on 0.5-5."""
    crowd = pd.DataFrame(
        {"member": ["a", "b", "c", "d"], "tmdb_id": [1, 1, 2, 2], "rating": [10, 10, 2, 2]}
    )
    bias = crowd_item_bias(crowd, prior=2.0)
    # mean 3.0 on the 0.5-5 scale; film 1: (10 - 2*3) / (2+2) = 1.0
    assert bias.loc[1] == pytest.approx(1.0)
    assert bias.loc[2] == pytest.approx(-1.0)


def test_the_pool_counts_distinct_members():
    crowd = keyed(
        [("a", "x", 5), ("b", "x", 5), ("c", "x", 5), ("a", "y", 5), ("a", "y2", 5)],
        {"x": 1, "y": 2, "y2": 2},
    )
    pool = build_pool(crowd, catalogue([(1, "X", 2000), (2, "Y", 2000)]), min_raters=2)
    assert pool["tmdb_id"].tolist() == [1]
    assert pool["raters"].tolist() == [3]


def test_the_pool_only_holds_recommendable_films():
    """An audience does not make an unreleased or delisted film recommendable."""
    crowd = keyed([("a", "x", 5), ("b", "x", 5)], {"x": 404})
    pool = build_pool(crowd, catalogue([(1, "X", 2000)]), min_raters=1)
    assert pool.empty
    assert pool.attrs["rated_elsewhere"] == 1


def test_crowd_less_share_counts_linked_but_unrated_films_as_cold():
    """links.csv lists films nobody in MovieLens rated; they have no item bias."""
    links = pd.DataFrame({"movieId": [1, 2], "tmdbId": [10.0, 20.0]})
    rated = rated_tmdb_ids(links, [1])
    assert rated == {10}

    cat = catalogue([(10, "A", 2000), (20, "B", 2000)], vote_count=[100, 0])
    pool = pd.DataFrame({"tmdb_id": [10, 20], "raters": [5, 5]})
    summary = pool_summary(pool, cat, rated)
    assert summary["movielens_cold"] == 0.5
    assert summary["zero_tmdb_votes"] == 1


def test_pool_by_year_reports_years_with_no_films():
    cat = catalogue([(1, "A", 2020), (2, "B", 2020), (3, "C", 2022)])
    pool = pd.DataFrame({"tmdb_id": [1]})
    report = pool_by_year(pool, cat, range(2020, 2024)).set_index("year")
    assert report.loc[2020, "pool"] == 1
    assert report.loc[2020, "share"] == "50.0%"
    assert report.loc[2022, "pool"] == 0
    assert report.loc[2023, "catalogue"] == 0
