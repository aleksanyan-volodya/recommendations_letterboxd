"""Tests for TMDb metadata enrichment.

Film and television payloads name the same ideas differently. These tests pin
the normalisation, and pin that the two fields everything else depends on --
``imdb_id`` (the join key to IMDb and MovieLens) and ``vote_count`` (the
popularity variable) -- survive it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from lbrec.enrich import enrich_films, fetch_one, normalize_movie, normalize_tv, popularity_report
from lbrec.tmdb import TmdbClient
from tests.test_tmdb import RecordingTransport, make_settings

MOVIE_PAYLOAD = {
    "id": 1398,
    "title": "Stalker",
    "original_title": "Сталкер",
    "original_language": "ru",
    "release_date": "1979-05-25",
    "runtime": 162,
    "status": "Released",
    "adult": False,
    "overview": "A guide leads two men through an area known as the Zone.",
    "tagline": "",
    "genres": [{"id": 878, "name": "Science Fiction"}, {"id": 18, "name": "Drama"}],
    "keywords": {"keywords": [{"id": 1, "name": "existentialism"}, {"id": 2, "name": "the zone"}]},
    "production_countries": [{"iso_3166_1": "SU"}, {"iso_3166_1": "DE"}],
    "spoken_languages": [{"iso_639_1": "ru"}],
    "vote_average": 8.1,
    "vote_count": 2500,
    "popularity": 17.5,
    "budget": 1000000,
    "revenue": 0,
    "belongs_to_collection": None,
    "external_ids": {"imdb_id": "tt0079944"},
    "credits": {
        "cast": [
            {"id": 10, "name": "Alexander Kaidanovsky", "order": 0},
            {"id": 11, "name": "Anatoly Solonitsyn", "order": 1},
        ],
        "crew": [
            {"id": 20, "name": "Andrei Tarkovsky", "job": "Director", "department": "Directing"},
            {"id": 21, "name": "Boris Strugatsky", "job": "Screenplay", "department": "Writing"},
            {
                "id": 22,
                "name": "Alexander Knyazhinsky",
                "job": "Director of Photography",
                "department": "Camera",
            },
        ],
    },
}

TV_PAYLOAD = {
    "id": 87108,
    "name": "Chernobyl",
    "original_name": "Chernobyl",
    "original_language": "en",
    "first_air_date": "2019-05-06",
    "episode_run_time": [60],
    "status": "Ended",
    "overview": "The true story of one of the worst man-made catastrophes.",
    "genres": [{"id": 18, "name": "Drama"}],
    "keywords": {"results": [{"id": 3, "name": "nuclear disaster"}]},
    "production_countries": [{"iso_3166_1": "GB"}],
    "spoken_languages": [{"iso_639_1": "en"}],
    "vote_average": 8.7,
    "vote_count": 4200,
    "popularity": 55.0,
    "created_by": [{"id": 30, "name": "Craig Mazin"}],
    "external_ids": {"imdb_id": "tt7366338"},
    "aggregate_credits": {"cast": [{"id": 40, "name": "Jared Harris", "order": 0}]},
}


def test_movie_normalisation_keeps_the_join_key_and_popularity():
    row = normalize_movie(MOVIE_PAYLOAD)
    assert row["imdb_id"] == "tt0079944"
    assert row["vote_count"] == 2500
    assert row["media_type"] == "movie"
    assert row["year"] == 1979


def test_movie_normalisation_flattens_nested_structures():
    row = normalize_movie(MOVIE_PAYLOAD)
    assert row["genres"] == ["Science Fiction", "Drama"]
    assert row["keywords"] == ["existentialism", "the zone"]
    assert row["production_countries"] == ["SU", "DE"]
    assert row["spoken_languages"] == ["ru"]


def test_crew_is_split_by_role():
    row = normalize_movie(MOVIE_PAYLOAD)
    assert row["directors"] == ["Andrei Tarkovsky"]
    assert row["director_ids"] == [20]
    assert row["writers"] == ["Boris Strugatsky"]
    assert row["cast"] == ["Alexander Kaidanovsky", "Anatoly Solonitsyn"]  # order respected


def test_tv_normalisation_maps_onto_the_film_schema():
    """Series use name/first_air_date/episode_run_time and nest keywords differently."""
    row = normalize_tv(TV_PAYLOAD)
    assert row["title"] == "Chernobyl"
    assert row["release_date"] == "2019-05-06"
    assert row["year"] == 2019
    assert row["runtime"] == 60
    assert row["keywords"] == ["nuclear disaster"]
    assert row["directors"] == ["Craig Mazin"]  # creators stand in for directors
    assert row["media_type"] == "tv"
    assert row["budget"] is None  # does not exist for television


def test_missing_optional_fields_do_not_crash():
    row = normalize_movie({"id": 1, "title": "Bare"})
    assert row["genres"] == [] and row["keywords"] == [] and row["cast"] == []
    assert row["imdb_id"] is None and row["vote_count"] is None


def test_imdb_id_falls_back_to_the_top_level_field():
    row = normalize_movie({"id": 1, "title": "x", "imdb_id": "tt0000001"})
    assert row["imdb_id"] == "tt0000001"


def test_fetch_one_uses_the_right_namespace(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/3/tv/"):
            return httpx.Response(200, json=TV_PAYLOAD)
        return httpx.Response(200, json=MOVIE_PAYLOAD)

    transport = RecordingTransport(handler)
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        assert fetch_one(client, 1398, "movie")["media_type"] == "movie"
        assert fetch_one(client, 87108, "tv")["media_type"] == "tv"
    assert transport.requests[0].url.path == "/3/movie/1398"
    assert transport.requests[1].url.path == "/3/tv/87108"


def test_fetch_one_returns_none_for_a_deleted_title(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(404, json={"status_code": 34}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        assert fetch_one(client, 999, "movie") is None


def test_enrich_fetches_a_shared_tmdb_id_only_once(tmp_path: Path):
    """Two Letterboxd entries can point at one TMDb record; store and fetch it once."""
    film_map = pd.DataFrame(
        [
            {"film_key": "aaaa", "tmdb_id": 1398, "media_type": "movie"},
            {"film_key": "bbbb", "tmdb_id": 1398, "media_type": "movie"},
            {"film_key": "cccc", "tmdb_id": pd.NA, "media_type": None},
        ]
    )
    transport = RecordingTransport(lambda r: httpx.Response(200, json=MOVIE_PAYLOAD))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        films = enrich_films(client, film_map)
    assert len(films) == 1
    assert len(transport.requests) == 1


def test_enrich_records_ids_tmdb_no_longer_has(tmp_path: Path):
    film_map = pd.DataFrame([{"film_key": "aaaa", "tmdb_id": 999, "media_type": "movie"}])
    transport = RecordingTransport(lambda r: httpx.Response(404, json={}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        films = enrich_films(client, film_map)
    assert films.empty
    assert films.attrs["missing_tmdb_ids"] == [999]


def test_popularity_report_splits_into_deciles():
    films = pd.DataFrame({"vote_count": list(range(100))})
    report = popularity_report(films)
    assert len(report) == 10
    assert report["films"].sum() == 100
    assert report.loc[0, "votes_min"] == 0
    assert report.loc[9, "votes_max"] == 99


def test_popularity_report_survives_an_empty_catalogue():
    assert popularity_report(pd.DataFrame({"vote_count": []})).empty
