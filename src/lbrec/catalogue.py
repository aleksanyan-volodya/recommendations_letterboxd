"""Build the candidate catalogue: the films we are allowed to recommend.

Until this exists the project cannot recommend anything -- every model so far
could only score the 1741 films already in the export. It also fixes a defect in
everything measured before it: "popularity decile" was being cut against the
user's *own library*, so "decile 1" meant "the least popular film I have already
watched", not "the least popular film in cinema".

The catalogue is assembled from the Letterboxd crowd dump rather than from the
TMDb API, because that dump already carries TMDb-derived metadata for ~285k
films. Enriching that many titles through the API would be ~13 hours of
requests; reading a CSV takes seconds.

**It is deliberately not filtered by popularity.** Trimming to "films with
enough votes" would be the easy way to make it small, and would silently
reintroduce exactly the bias this project exists to remove: films could not be
recommended precisely because they are obscure.

What the dump does *not* carry is keywords, cast and crew -- the features the
content models lean on hardest. Hence two stages:

1. **Retrieval** over the whole catalogue, using only features the catalogue
   actually has, with the model trained on that same restricted set so training
   and inference see the same world.
2. **Reranking** of the top candidates after enriching just those through the
   API, where the full feature set is available.

Scoring 285k films with a model trained on keywords they do not have would be a
train/inference mismatch dressed up as a recommendation.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns the dump provides that our feature builder understands.
CATALOGUE_COLUMNS = [
    "tmdb_id",
    "imdb_id",
    "title",
    "year",
    "release_date",
    "runtime",
    "overview",
    "genres",
    "production_countries",
    "original_language",
    "vote_average",
    "vote_count",
    "media_type",
]


def _json_list(raw: str) -> list[str]:
    """Parse a JSON array cell, tolerating the malformed ones."""
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _number(raw: str, *, integer: bool = False):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return int(value) if integer else value


def read_letterboxd_dump(zip_path: Path, member: str = "movie_data.csv") -> pd.DataFrame:
    """Read the crowd dump's film table into the catalogue schema.

    Parsed with the ``csv`` module row by row rather than with pandas or
    pyarrow: several hundred rows are truncated mid-record, which overflows
    pandas' C parser and makes pyarrow raise. Rows whose field count does not
    match the header are skipped and counted rather than guessed at.
    """
    csv.field_size_limit(sys.maxsize)
    rows: list[dict] = []
    skipped = 0

    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}

        for record in reader:
            if len(record) != len(header):
                skipped += 1
                continue
            tmdb_id = _number(record[index["tmdb_id"]], integer=True)
            title = record[index["movie_title"]].strip()
            if tmdb_id is None or not title:
                skipped += 1
                continue

            runtime = _number(record[index["runtime"]], integer=True)
            rows.append(
                {
                    "tmdb_id": tmdb_id,
                    "imdb_id": record[index["imdb_id"]].strip() or None,
                    "title": title,
                    "year": _number(record[index["year_released"]], integer=True),
                    "release_date": record[index["release_date"]].strip() or None,
                    # The dump writes 0 for "unknown", which is not a runtime.
                    "runtime": runtime if runtime else None,
                    "overview": record[index["overview"]].strip(),
                    "genres": _json_list(record[index["genres"]]),
                    "production_countries": _json_list(record[index["production_countries"]]),
                    "original_language": record[index["original_language"]].strip() or None,
                    "vote_average": _number(record[index["vote_average"]]),
                    "vote_count": _number(record[index["vote_count"]], integer=True),
                    "media_type": "movie",
                }
            )

    frame = pd.DataFrame.from_records(rows, columns=CATALOGUE_COLUMNS)
    frame = frame.dropna(subset=["tmdb_id"]).drop_duplicates("tmdb_id", ignore_index=True)
    for column in ("tmdb_id", "year", "runtime", "vote_count"):
        frame[column] = frame[column].astype("Int64")
    frame["vote_average"] = frame["vote_average"].astype("Float64")
    frame.attrs["skipped_rows"] = skipped
    return frame


def merge_known_films(catalogue: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """Overlay full TMDb metadata where we already have it.

    Films already enriched carry keywords, cast and crew; the rest do not. The
    overlay keeps the richer record so the catalogue is as good as the data
    allows, while `shared_feature_columns` still decides what a retrieval model
    is permitted to look at.
    """
    enriched = enriched[enriched["media_type"] == "movie"].copy()
    enriched["tmdb_id"] = enriched["tmdb_id"].astype("Int64")
    catalogue = catalogue[~catalogue["tmdb_id"].isin(set(enriched["tmdb_id"].dropna()))]
    merged = pd.concat([enriched, catalogue], ignore_index=True)
    merged["enriched"] = merged.index < len(enriched)
    return merged


def exclude_seen(catalogue: pd.DataFrame, seen_tmdb_ids: set[int]) -> pd.DataFrame:
    """Drop films the user has already watched.

    ``seen`` must come from ``film_status``, which is the union of watched,
    rated and diarised -- not ``watched.csv`` alone, which under-counts when
    Letterboxd deletes a film entry.
    """
    ids = catalogue["tmdb_id"].astype("Int64")
    return catalogue[~ids.isin(seen_tmdb_ids)].reset_index(drop=True)


def catalogue_report(catalogue: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Popularity deciles of the real catalogue.

    These are the boundaries every later popularity claim should be measured
    against. Cutting them against a user's own library instead -- which is what
    happened before this module existed -- makes "decile 1" mean something
    entirely different and much less obscure.
    """
    votes = catalogue["vote_count"].dropna().astype(float)
    if votes.empty:
        return pd.DataFrame(columns=["decile", "films", "votes_min", "votes_max"])

    labels = pd.qcut(votes.rank(method="first"), q=bins, labels=range(1, bins + 1))
    return (
        pd.DataFrame({"decile": labels, "votes": votes})
        .groupby("decile", observed=True)
        .agg(films=("votes", "size"), votes_min=("votes", "min"), votes_max=("votes", "max"))
        .reset_index()
        .astype({"votes_min": int, "votes_max": int})
    )
