"""Tests for the MovieLens loader.

The join through ``links.csv`` has one trap that silently produces zero matches,
and the coverage measurement has to be sliced by popularity to say anything
useful. Both are pinned here.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from lbrec.movielens import (
    archive_prefix,
    by_movie_id,
    by_tmdb_id,
    coverage_by_popularity,
    imdb_tt,
    item_bias,
    link_films,
    load,
    prepare,
)

RATINGS = "userId,movieId,rating,timestamp\n1,1,4.0,1225734739\n1,2,3.5,1225865086\n2,1,5.0,1000\n"
MOVIES = (
    "movieId,title,genres\n1,Toy Story (1995),Adventure|Animation\n2,Jumanji (1995),Adventure\n"
)
LINKS = "movieId,imdbId,tmdbId\n1,114709,862\n2,113497,8844\n3,99999999,\n"
TAGS = "userId,movieId,tag,timestamp\n1,1,pixar,1225734739\n"


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "ml-test.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ml-32m/ratings.csv", RATINGS)
        z.writestr("ml-32m/movies.csv", MOVIES)
        z.writestr("ml-32m/links.csv", LINKS)
        z.writestr("ml-32m/tags.csv", TAGS)
        z.writestr("ml-32m/README.txt", "hello")
    return path


def test_imdb_ids_are_restored_to_tt_form():
    """MovieLens strips the tt prefix and leading zeros; joining raw matches nothing."""
    restored = imdb_tt(pd.Series([114709, 113497, 79944, None]))
    assert restored.tolist()[:3] == ["tt0114709", "tt0113497", "tt0079944"]
    assert pd.isna(restored.iloc[3])


def test_imdb_ids_longer_than_seven_digits_are_not_truncated():
    assert imdb_tt(pd.Series([12345678])).iloc[0] == "tt12345678"


def test_archive_prefix_is_discovered_not_assumed():
    """ml-25m and ml-32m use different top-level directories."""

    class FakeZip:
        def namelist(self):
            return ["ml-25m/links.csv", "ml-25m/movies.csv"]

    assert archive_prefix(FakeZip()) == "ml-25m"


def test_archive_prefix_rejects_an_ambiguous_archive():
    class FakeZip:
        def namelist(self):
            return ["a/x.csv", "b/y.csv"]

    with pytest.raises(ValueError, match="one top-level directory"):
        archive_prefix(FakeZip())


def test_prepare_converts_members_to_parquet(archive: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    written = prepare(archive, out)
    assert set(written) == {"ratings", "movies", "links", "tags"}
    ratings = load(out, "ratings")
    assert len(ratings) == 3
    assert ratings["rating"].tolist() == [4.0, 3.5, 5.0]


def test_prepare_skips_members_already_converted(archive: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    prepare(archive, out)
    stamp = (out / "links.parquet").stat().st_mtime_ns
    prepare(archive, out)
    assert (out / "links.parquet").stat().st_mtime_ns == stamp


def test_prepare_tolerates_absent_members(archive: Path, tmp_path: Path):
    """ml-32m ships no genome files; asking for them must not fail."""
    written = prepare(archive, tmp_path / "prepared", members=["links", "genome-scores"])
    assert set(written) == {"links"}


def test_link_films_joins_on_tmdb_id(archive: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    prepare(archive, out)
    film_map = pd.DataFrame(
        [
            {"film_key": "aaaa", "tmdb_id": 862, "media_type": "movie"},
            {"film_key": "bbbb", "tmdb_id": 999999, "media_type": "movie"},
            {"film_key": "cccc", "tmdb_id": pd.NA, "media_type": None},
        ]
    )
    linked = link_films(film_map, load(out, "links")).set_index("film_key")
    assert linked.loc["aaaa", "in_movielens"]
    assert not linked.loc["bbbb", "in_movielens"]
    assert "cccc" not in linked.index  # unresolved films are not carried through


def test_link_films_ignores_rows_with_no_tmdb_id(archive: Path, tmp_path: Path):
    """links.csv has rows with a blank tmdbId; they must not collapse into one key."""
    out = tmp_path / "prepared"
    prepare(archive, out)
    film_map = pd.DataFrame([{"film_key": "aaaa", "tmdb_id": 862, "media_type": "movie"}])
    linked = link_films(film_map, load(out, "links"))
    assert len(linked) == 1  # no fan-out from the null-tmdbId row


def test_rekeying_by_movie_id_never_fans_out():
    """links.csv repeats and blanks tmdbIds; one value must land on one film."""
    links = pd.DataFrame({"movieId": [1, 2, 3, 4], "tmdbId": [10.0, 10.0, None, 30.0]})
    out = by_movie_id(pd.Series({10: 0.5, 30: -0.2, 99: 1.0}), links)
    assert out.to_dict() == {1: 0.5, 4: -0.2}


def test_rekeying_by_tmdb_id_is_the_inverse():
    links = pd.DataFrame({"movieId": [1, 2, 3], "tmdbId": [10.0, 10.0, 30.0]})
    out = by_tmdb_id(pd.Series({1: 0.5, 2: 9.9, 3: -0.2}), links)
    assert out.to_dict() == {10: 0.5, 30: -0.2}


def test_item_bias_is_the_models_shrunk_estimator():
    ratings = pd.DataFrame({"movieId": [1, 1, 2, 2], "rating": [5.0, 5.0, 1.0, 1.0]})
    bias = item_bias(ratings, prior=2.0)
    # mean 3.0; film 1: (10 - 2*3) / (2+2) = 1.0
    assert bias.to_dict() == {1: 1.0, 2: -1.0}


def test_coverage_is_reported_per_popularity_decile():
    """The headline number hides whether the tail is covered at all."""
    catalogue = pd.DataFrame(
        {
            "tmdb_id": pd.Series(range(100), dtype="Int64"),
            "vote_count": pd.Series(range(100), dtype="Int64"),
        }
    )
    linked = pd.DataFrame(
        {
            "tmdb_id": pd.Series(range(100), dtype="Int64"),
            # covered only in the top half of the catalogue
            "in_movielens": [i >= 50 for i in range(100)],
            "movieId": pd.Series(range(100), dtype="Int64"),
        }
    )
    report = coverage_by_popularity(linked, catalogue).set_index("decile")
    assert report.loc[1, "coverage"] == "0.0%"
    assert report.loc[10, "coverage"] == "100.0%"


def test_coverage_can_include_ratings_depth():
    catalogue = pd.DataFrame(
        {
            "tmdb_id": pd.Series([1, 2], dtype="Int64"),
            "vote_count": pd.Series([10, 20000], dtype="Int64"),
        }
    )
    linked = pd.DataFrame(
        {
            "tmdb_id": pd.Series([1, 2], dtype="Int64"),
            "in_movielens": [True, True],
            "movieId": pd.Series([10, 20], dtype="Int64"),
        }
    )
    per_movie = pd.Series({10: 3, 20: 5000})
    report = coverage_by_popularity(linked, catalogue, per_movie)
    assert "median_ml_ratings" in report.columns
    assert report["median_ml_ratings"].tolist() == [3, 5000]
