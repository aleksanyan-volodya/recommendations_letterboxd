"""Fetch full TMDb metadata for every resolved film.

One request per film, with genres, keywords, credits and external IDs appended,
so the whole content feature set arrives in a single round trip each. Responses
are cached, so a re-run costs nothing.

Two fields matter out of proportion to the rest:

``imdb_id``
    The join key to the IMDb bulk datasets and to MovieLens ``links.csv``.
    Without it the external data cannot be attached at all.
``vote_count``
    The popularity variable. Everything in the debiasing plan is a function
    of it, so it is collected here and kept as a first-class column from the
    start rather than bolted on later.

Film and television use different field names for the same ideas. Both are
normalised onto the film schema here so downstream code sees one shape; TV rows
stay tagged with ``media_type`` so they can be kept out of the candidate
catalogue while still informing the user model.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from lbrec.resolve import MEDIA_MOVIE, MEDIA_TV
from lbrec.tmdb import TmdbClient

#: Everything we need per title, in one request.
MOVIE_APPEND = "external_ids,keywords,credits"
TV_APPEND = "external_ids,keywords,aggregate_credits"

#: How many billed cast members to keep. Beyond roughly this depth the credits
#: stop saying anything about why a film was made the way it was.
CAST_DEPTH = 10

WRITING_JOBS = frozenset({"Writer", "Screenplay", "Story", "Author", "Novel"})

FILM_COLUMNS = [
    "tmdb_id",
    "media_type",
    "imdb_id",
    "title",
    "original_title",
    "original_language",
    "release_date",
    "year",
    "runtime",
    "status",
    "adult",
    "overview",
    "tagline",
    "genres",
    "keywords",
    "production_countries",
    "spoken_languages",
    "vote_average",
    "vote_count",
    "popularity",
    "budget",
    "revenue",
    "collection_id",
    "collection_name",
    "director_ids",
    "directors",
    "writer_ids",
    "writers",
    "cast_ids",
    "cast",
]


def _names(items: list[dict] | None, key: str = "name") -> list[str]:
    return [item[key] for item in (items or []) if item.get(key)]


def _year(release_date: str | None) -> int | None:
    try:
        return int((release_date or "")[:4])
    except (TypeError, ValueError):
        return None


def _crew_by_role(crew: list[dict]) -> tuple[list[dict], list[dict]]:
    directors = [person for person in crew if person.get("job") == "Director"]
    writers = [
        person
        for person in crew
        if person.get("job") in WRITING_JOBS or person.get("department") == "Writing"
    ]
    return directors, writers


def normalize_movie(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a TMDb film payload into the feature schema."""
    credits = payload.get("credits") or {}
    directors, writers = _crew_by_role(credits.get("crew") or [])
    cast = sorted(
        credits.get("cast") or [], key=lambda person: person.get("order", CAST_DEPTH * 10)
    )[:CAST_DEPTH]
    collection = payload.get("belongs_to_collection") or {}
    release_date = payload.get("release_date") or None

    return {
        "tmdb_id": payload.get("id"),
        "media_type": MEDIA_MOVIE,
        "imdb_id": (payload.get("external_ids") or {}).get("imdb_id")
        or payload.get("imdb_id")
        or None,
        "title": payload.get("title") or "",
        "original_title": payload.get("original_title") or "",
        "original_language": payload.get("original_language") or None,
        "release_date": release_date,
        "year": _year(release_date),
        "runtime": payload.get("runtime"),
        "status": payload.get("status") or None,
        "adult": bool(payload.get("adult")),
        "overview": payload.get("overview") or "",
        "tagline": payload.get("tagline") or "",
        "genres": _names(payload.get("genres")),
        "keywords": _names((payload.get("keywords") or {}).get("keywords")),
        "production_countries": [
            country["iso_3166_1"]
            for country in (payload.get("production_countries") or [])
            if country.get("iso_3166_1")
        ],
        "spoken_languages": [
            language["iso_639_1"]
            for language in (payload.get("spoken_languages") or [])
            if language.get("iso_639_1")
        ],
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "popularity": payload.get("popularity"),
        "budget": payload.get("budget"),
        "revenue": payload.get("revenue"),
        "collection_id": collection.get("id"),
        "collection_name": collection.get("name"),
        "director_ids": [person["id"] for person in directors if person.get("id")],
        "directors": _names(directors),
        "writer_ids": [person["id"] for person in writers if person.get("id")],
        "writers": _names(writers),
        "cast_ids": [person["id"] for person in cast if person.get("id")],
        "cast": _names(cast),
    }


def normalize_tv(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a TMDb series payload onto the same schema.

    Series use ``name``/``first_air_date``/``episode_run_time``, put keywords
    under ``results``, and credit creators rather than directors. Budget and
    revenue do not exist for television and stay null.
    """
    credits = payload.get("aggregate_credits") or {}
    cast = sorted(
        credits.get("cast") or [], key=lambda person: person.get("order", CAST_DEPTH * 10)
    )[:CAST_DEPTH]
    creators = payload.get("created_by") or []
    runtimes = payload.get("episode_run_time") or []
    first_air_date = payload.get("first_air_date") or None

    return {
        "tmdb_id": payload.get("id"),
        "media_type": MEDIA_TV,
        "imdb_id": (payload.get("external_ids") or {}).get("imdb_id") or None,
        "title": payload.get("name") or "",
        "original_title": payload.get("original_name") or "",
        "original_language": payload.get("original_language") or None,
        "release_date": first_air_date,
        "year": _year(first_air_date),
        "runtime": runtimes[0] if runtimes else None,
        "status": payload.get("status") or None,
        "adult": bool(payload.get("adult")),
        "overview": payload.get("overview") or "",
        "tagline": payload.get("tagline") or "",
        "genres": _names(payload.get("genres")),
        "keywords": _names((payload.get("keywords") or {}).get("results")),
        "production_countries": [
            country["iso_3166_1"]
            for country in (payload.get("production_countries") or [])
            if country.get("iso_3166_1")
        ],
        "spoken_languages": [
            language["iso_639_1"]
            for language in (payload.get("spoken_languages") or [])
            if language.get("iso_639_1")
        ],
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "popularity": payload.get("popularity"),
        "budget": None,
        "revenue": None,
        "collection_id": None,
        "collection_name": None,
        "director_ids": [person["id"] for person in creators if person.get("id")],
        "directors": _names(creators),
        "writer_ids": [],
        "writers": [],
        "cast_ids": [person["id"] for person in cast if person.get("id")],
        "cast": _names(cast),
    }


def fetch_one(client: TmdbClient, tmdb_id: int, media_type: str) -> dict[str, Any] | None:
    """Fetch and normalise one title. ``None`` if TMDb no longer has it."""
    if media_type == MEDIA_TV:
        payload = client.tv(tmdb_id, append=TV_APPEND)
        return normalize_tv(payload) if payload else None
    payload = client.movie(tmdb_id, append=MOVIE_APPEND)
    return normalize_movie(payload) if payload else None


def enrich_films(client: TmdbClient, film_map: pd.DataFrame, *, progress=None) -> pd.DataFrame:
    """Fetch metadata for every resolved title in ``film_map``.

    Deduplicated by ``(tmdb_id, media_type)``: two Letterboxd entries may point
    at one TMDb record, and it should still only be fetched and stored once.
    """
    resolved = film_map[film_map["tmdb_id"].notna()][["tmdb_id", "media_type"]]
    unique = resolved.drop_duplicates().itertuples(index=False)

    records, missing = [], []
    for row in unique:
        record = fetch_one(client, int(row.tmdb_id), row.media_type)
        if record is None:
            missing.append(int(row.tmdb_id))
        else:
            records.append(record)
        if progress is not None:
            progress.update(1)

    frame = pd.DataFrame.from_records(records, columns=FILM_COLUMNS)
    for column in (
        "tmdb_id",
        "year",
        "runtime",
        "vote_count",
        "budget",
        "revenue",
        "collection_id",
    ):
        frame[column] = frame[column].astype("Int64")
    for column in ("vote_average", "popularity"):
        frame[column] = frame[column].astype("Float64")
    frame.attrs["missing_tmdb_ids"] = missing
    return frame


def popularity_report(films: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Distribution of the catalogue across popularity deciles.

    Printed on every enrich run because popularity is the axis this project is
    ultimately judged on: if the library is entirely in the top deciles, there
    is no long tail here to surface, and that needs to be visible early.
    """
    films = films[films["vote_count"].notna()]
    if films.empty:
        return pd.DataFrame(columns=["decile", "films", "votes_min", "votes_max"])

    votes = films["vote_count"].astype("int64")
    labels = pd.qcut(votes.rank(method="first"), q=bins, labels=range(1, bins + 1))
    report = (
        pd.DataFrame({"decile": labels, "votes": votes})
        .groupby("decile", observed=True)
        .agg(films=("votes", "size"), votes_min=("votes", "min"), votes_max=("votes", "max"))
        .reset_index()
    )
    return report
