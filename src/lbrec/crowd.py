"""The Letterboxd crowd: rating dumps scraped from its most-followed members.

This is the second crowd signal after MovieLens, and the one that decides what
may be recommended at all.

**What a dump establishes.** Letterboxd takes all of its film data from TMDb and
imports any TMDb film on demand, so the TMDb catalogue already *is* what exists
on Letterboxd. A dump says something narrower and more useful: which films have
an audience among people who watch a great deal. The candidate pool built here
is therefore an audience floor, not an existence check.

**The dumps.** Each one holds every rated film of each member, scraped from the
popular-members pages:

==========  ======================  ======  =================================
dump        scraped                 scale   film identity
==========  ======================  ======  =================================
samlearner  Nov 2020 -> Mar 2022    1-10    slug, plus Letterboxd's TMDb link
freeth      Oct 2023                0.5-5   slug, title and year only
==========  ======================  ======  =================================

samlearner is a rolling scrape rather than a snapshot: its Mongo ObjectIds put
38% of ratings after 2020, so its horizon is March 2022, not late 2020.

Three traps, each pinned by a test:

1. Slugs ``null`` and ``nan`` are real films. pandas' default NA parsing turns
   them into missing values and they silently vanish.
2. A dump holds a member's *whole* rated list as of one moment, so a member's
   newer snapshot replaces the older one wholesale. Merging per (member, film)
   would resurrect ratings the member has since deleted or changed.
3. The scales differ. Everything is stored as 1-10 integers, which is exact
   for half stars.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from lbrec.letterboxd import normalize_title

DUMP_SAMLEARNER = "samlearner"
DUMP_FREETH = "freeth"

RATING_COLUMNS = ["member", "slug", "rating"]
FILM_COLUMNS = ["slug", "tmdb_id", "title", "year"]

#: How a slug got its TMDb id, most to least direct.
LINKED = "letterboxd"  # the dump carries Letterboxd's own TMDb link
MATCHED = "title_year"  # exactly one catalogue film has this title and year
AMBIGUOUS = "ambiguous"  # several do; left unmapped rather than guessed
UNMATCHED = "unmatched"  # none does, or the slug has no year
DEAD_LINK = "dead link"  # linked to an id TMDb has since deleted, and no match
NO_RECORD = "no film record"  # rated, but in no dump's film table

#: A film needs this many distinct members' ratings to enter the pool.
#:
#: The floor is a share of the crowd, not an absolute count: 10 of the merged
#: dumps' 14,826 members is the same 0.067% as the 5 of 7,477 the pool was
#: first designed with. Kept at 5, the larger crowd admitted a deeper tail --
#: 141k films, 52% unrated by MovieLens, outside the range the stratified
#: ranker was validated on. At 10: ~95k films, 38% unrated by MovieLens, a
#: median of 24 TMDb votes. Revisit if the crowd changes size.
MIN_RATERS = 10


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def _read_ratings(raw, *, member: str, slug: str, rating: str, rating_type) -> pd.DataFrame:
    table = pa_csv.read_csv(
        raw,
        convert_options=pa_csv.ConvertOptions(
            include_columns=[member, slug, rating],
            column_types={member: pa.string(), slug: pa.string(), rating: rating_type},
            # Explicit, because this is the whole of trap 1: "null" is a slug.
            strings_can_be_null=False,
        ),
    )
    frame = table.to_pandas().rename(columns={member: "member", slug: "slug", rating: "rating"})
    # Usernames are case-insensitive on Letterboxd; the two scrapers need not agree.
    frame["member"] = frame["member"].str.strip().str.casefold()
    return frame[RATING_COLUMNS]


def _ten_point(stars: pd.Series) -> pd.Series:
    """Map a rating onto 1-10 integers, refusing anything off the scale."""
    values = pd.to_numeric(stars, errors="coerce")
    if values.isna().any() or not values.between(1, 10).all() or (values % 1 != 0).any():
        bad = stars[values.isna() | ~values.between(1, 10) | (values % 1 != 0)].unique()[:5]
        raise ValueError(f"ratings off the 1-10 scale: {list(bad)}")
    return values.astype("int8")


def read_samlearner_ratings(zip_path: Path) -> pd.DataFrame:
    """``ratings_export.csv``: one row per (member, film), already on 1-10."""
    with zipfile.ZipFile(zip_path) as archive, archive.open("ratings_export.csv") as raw:
        frame = _read_ratings(
            raw, member="user_id", slug="movie_id", rating="rating_val", rating_type=pa.int16()
        )
    frame["rating"] = _ten_point(frame["rating"])
    return frame


def read_samlearner_films(zip_path: Path) -> pd.DataFrame:
    """``movie_data.csv``: slug to Letterboxd's own TMDb link, with title and year.

    Parsed with the ``csv`` module because several hundred rows are truncated
    mid-record, which pyarrow refuses outright. Rows whose field count does not
    match the header are skipped and counted rather than guessed at.
    """
    csv.field_size_limit(sys.maxsize)
    rows: list[tuple[str, str, str, str]] = []
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive, archive.open("movie_data.csv") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        fields = [index[name] for name in ("movie_id", "tmdb_id", "movie_title", "year_released")]
        for record in reader:
            if len(record) != len(header):
                skipped += 1
                continue
            rows.append(tuple(record[i] for i in fields))

    frame = pd.DataFrame(rows, columns=FILM_COLUMNS)
    frame["tmdb_id"] = pd.to_numeric(frame["tmdb_id"], errors="coerce").astype("Int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame = frame.drop_duplicates("slug", ignore_index=True)
    frame.attrs["skipped_rows"] = skipped
    return frame


def read_freeth_ratings(zip_path: Path) -> pd.DataFrame:
    """``ratings.csv``: stars on 0.5-5, doubled onto the 1-10 scale."""
    with zipfile.ZipFile(zip_path) as archive, archive.open("ratings.csv") as raw:
        frame = _read_ratings(
            raw, member="user_name", slug="film_id", rating="rating", rating_type=pa.float32()
        )
    frame["rating"] = _ten_point((frame["rating"] * 2).round())
    return frame


def read_freeth_films(zip_path: Path) -> pd.DataFrame:
    """``films.csv``: slug, title and year. No TMDb id."""
    with zipfile.ZipFile(zip_path) as archive, archive.open("films.csv") as raw:
        frame = pd.read_csv(
            raw, usecols=["film_id", "film_name", "year"], dtype=str, keep_default_na=False
        )
    frame = frame.rename(columns={"film_id": "slug", "film_name": "title"})
    frame["tmdb_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame[FILM_COLUMNS].drop_duplicates("slug", ignore_index=True)


# --------------------------------------------------------------------------
# merging the dumps
# --------------------------------------------------------------------------
def latest_snapshot(dumps: Sequence[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Merge dumps, keeping each member's ratings from the newest dump only.

    ``dumps`` is ordered oldest first. A member who appears in a newer dump is
    taken entirely from it: that scrape is their whole rated list as of a later
    date, so a film missing from it was un-rated, not forgotten. Members who
    dropped off the popular lists keep their older snapshot -- their ratings
    are stale but not wrong.
    """
    kept: list[pd.DataFrame] = []
    claimed: set[str] = set()
    superseded: dict[str, int] = {}
    for name, ratings in reversed(dumps):
        members = set(ratings["member"].unique())
        superseded[name] = len(members & claimed)
        kept.append(ratings[~ratings["member"].isin(claimed)].assign(dump=name))
        claimed |= members

    merged = pd.concat(kept[::-1], ignore_index=True)
    merged["dump"] = merged["dump"].astype("category")
    merged.attrs["superseded"] = superseded
    return merged


# --------------------------------------------------------------------------
# slugs to TMDb ids
# --------------------------------------------------------------------------
def _normalised(titles: pd.Series) -> pd.Series:
    return titles.fillna("").astype(str).map(normalize_title)


def match_by_title_year(unlinked: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Give a TMDb id to slugs without one, when exactly one catalogue film fits.

    A slug matches a catalogue film when its normalised title equals the film's
    normalised title or original title, and their years are within one. An
    exact year is preferred over an adjacent one; beyond that, more than one
    candidate means the slug stays unmapped.

    Titles must match literally. The user-export resolver also strips articles,
    because TMDb search results can differ from what a person typed; here the
    slug's title *is* TMDb's title as Letterboxd copied it, so a variant only
    adds false matches. It did: "Education" (2020), a Small Axe film TMDb has
    since moved to television, matched "The Education" (2020) and took its 654
    raters.

    The user-export resolver also breaks ties by TMDb popularity, which is right
    there: a person's "Parasite (2019)" is almost surely the famous one. It is
    wrong here. This table decides which films have an audience, and handing an
    ambiguous film's raters to its better-known namesake inflates the head of
    the catalogue by construction.
    """
    out = unlinked[["slug"]].copy()
    out["tmdb_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["how"] = UNMATCHED
    if unlinked.empty:
        return out

    queries = unlinked[["slug", "year"]].assign(form=_normalised(unlinked["title"]))
    queries = queries.dropna(subset=["year"])
    queries = queries[queries["form"] != ""]
    wanted = set(queries["form"])

    films = catalogue[["tmdb_id", "year"]]
    forms = pd.concat(
        [
            films.assign(form=_normalised(catalogue[column]))
            for column in ("title", "original_title")
            if column in catalogue
        ]
    )
    forms = forms[forms["form"].isin(wanted)].dropna(subset=["year"])

    pairs = queries.merge(forms, on="form", suffixes=("", "_tmdb"))
    pairs["delta"] = (pairs["year"] - pairs["year_tmdb"]).abs()
    pairs = pairs[pairs["delta"] <= 1].drop_duplicates(["slug", "tmdb_id"])

    # Closest year first; the slug resolves only if that tier holds one film.
    closest = pairs[pairs["delta"] == pairs.groupby("slug")["delta"].transform("min")]
    per_slug = closest.groupby("slug")["tmdb_id"].agg(["nunique", "first"])
    unique = per_slug[per_slug["nunique"] == 1]["first"]
    ambiguous = per_slug.index[per_slug["nunique"] > 1]

    out = out.set_index("slug")
    out.loc[unique.index, "tmdb_id"] = unique.astype("Int64")
    out.loc[unique.index, "how"] = MATCHED
    out.loc[ambiguous, "how"] = AMBIGUOUS
    return out.reset_index()


def map_slugs(
    films: Sequence[pd.DataFrame], catalogue: pd.DataFrame, *, live_ids: set[int] | None = None
) -> pd.DataFrame:
    """One TMDb id per slug: Letterboxd's own link where a dump has it, else a match.

    ``films`` are the dumps' film tables, oldest first. A slug's own TMDb link is
    trusted even when its title disagrees with the catalogue's: checked against
    the catalogue, 95% agree on title and year, and nearly all the rest are
    translations or later retitles ("Deadpool 3" -> "Deadpool & Wolverine").
    Title and year come from the newest dump that mentions the slug.

    A link is *dead* when TMDb no longer has that id at all (``live_ids``: every
    id in TMDb's current export, video releases included). TMDb merged or
    deleted the entry and Letterboxd will have relinked the page, so the slug is
    matched by title and year like an unlinked one. A live id that is merely
    outside the catalogue -- a video release, an unreleased film -- is kept:
    rematching it could hand its audience to a namesake.
    """
    stacked = pd.concat(films, ignore_index=True)
    linked = stacked.dropna(subset=["tmdb_id"]).drop_duplicates("slug")[["slug", "tmdb_id"]]
    described = stacked.drop_duplicates("slug", keep="last")[["slug", "title", "year"]]

    table = described.merge(linked, on="slug", how="left")
    table["how"] = np.where(table["tmdb_id"].notna(), LINKED, UNMATCHED)
    if live_ids is not None:
        dead = table["tmdb_id"].notna() & ~table["tmdb_id"].isin(live_ids)
        table.loc[dead, "tmdb_id"] = pd.NA
        table.loc[dead, "how"] = DEAD_LINK

    matched = match_by_title_year(table[table["tmdb_id"].isna()], catalogue).set_index("slug")
    found = matched.index[matched["how"] != UNMATCHED]
    table = table.set_index("slug")
    table.loc[found, "tmdb_id"] = matched.loc[found, "tmdb_id"]
    table.loc[found, "how"] = matched.loc[found, "how"]
    return table.reset_index()[["slug", "title", "year", "tmdb_id", "how"]]


def mapping_report(
    ratings: pd.DataFrame, slug_map: pd.DataFrame, *, min_raters: int = MIN_RATERS
) -> pd.DataFrame:
    """What each mapping route carries: slugs, ratings, and would-be pool films.

    Counting slugs alone hides the cost, and counting ratings alone hides it the
    other way: unmapped slugs are mostly thinly rated, i.e. tail films, which
    is exactly the population this project exists for. ``pool_sized`` counts the
    slugs with enough raters that losing them shrinks the pool.
    """
    per_slug = ratings.groupby("slug")["member"].nunique().rename("raters").reset_index()
    joined = per_slug.merge(slug_map[["slug", "how"]], on="slug", how="left")
    joined["how"] = joined["how"].fillna(NO_RECORD)
    report = (
        joined.assign(pool_sized=joined["raters"] >= min_raters)
        .groupby("how")
        .agg(slugs=("slug", "size"), ratings=("raters", "sum"), pool_sized=("pool_sized", "sum"))
        .sort_values("ratings", ascending=False)
        .reset_index()
    )
    report["share"] = (report["ratings"] / report["ratings"].sum()).map("{:.2%}".format)
    return report


def ratings_by_film(ratings: pd.DataFrame, slug_map: pd.DataFrame) -> pd.DataFrame:
    """Key merged ratings by TMDb id, one row per (member, film).

    Several slugs can carry one TMDb id -- a renamed page keeps its old slug in
    older scrapes, and a few distinct cuts share an id. A member who rated two of
    them is still one rater; their ratings are averaged and rounded.
    """
    keyed = ratings.merge(slug_map[["slug", "tmdb_id"]], on="slug", how="left")
    keyed = keyed.dropna(subset=["tmdb_id"])
    keyed["tmdb_id"] = keyed["tmdb_id"].astype("int64")

    doubled = keyed.duplicated(["member", "tmdb_id"], keep=False)
    if doubled.any():
        merged = (
            keyed[doubled]
            .groupby(["member", "tmdb_id"], as_index=False, observed=True)
            .agg(rating=("rating", "mean"), dump=("dump", "last"))
        )
        merged["rating"] = merged["rating"].round().astype("int8")
        keyed = pd.concat([keyed[~doubled], merged], ignore_index=True)

    out = keyed[["member", "tmdb_id", "rating", "dump"]].reset_index(drop=True)
    out["dump"] = out["dump"].astype("category")
    out.attrs["merged_duplicates"] = int(doubled.sum())
    return out


def as_model_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """The crowd in the shape every `generalise` model reads.

    ``userId`` is an integer code per member (strings would cost gigabytes over
    20M rows), ``movieId`` is the TMDb id, and ``rating`` is halved onto the
    0.5-5 scale the models clip to -- the scale a person's own export uses.
    """
    codes, _ = pd.factorize(ratings["member"])
    return pd.DataFrame(
        {
            "userId": codes.astype("int32"),
            "movieId": ratings["tmdb_id"].astype("int64").to_numpy(),
            "rating": (ratings["rating"].astype("float32") / 2).to_numpy(),
        }
    )


def crowd_item_bias(ratings: pd.DataFrame, *, prior: float = 20.0) -> pd.Series:
    """Each film's shrunk departure from the crowd's mean, on the 0.5-5 scale.

    The same estimator the models use for their own crowd, so a bias borrowed
    from here means the same kind of thing: ``(sum - n * mean) / (n + prior)``.
    Halved from the stored 1-10 scale to match the models' rating scale.
    """
    stars = ratings["rating"].astype("float64") / 2.0
    mean = stars.mean()
    grouped = stars.groupby(ratings["tmdb_id"]).agg(["sum", "count"])
    bias = (grouped["sum"] - grouped["count"] * mean) / (grouped["count"] + prior)
    return bias.rename("bias")


# --------------------------------------------------------------------------
# the pool
# --------------------------------------------------------------------------
def build_pool(
    ratings: pd.DataFrame, catalogue: pd.DataFrame, *, min_raters: int = MIN_RATERS
) -> pd.DataFrame:
    """Recommendable films rated by at least ``min_raters`` distinct members.

    Restricted to the catalogue, so a film the crowd rated but TMDb no longer
    lists as released cannot be recommended because it has an audience.
    """
    raters = ratings.groupby("tmdb_id")["member"].nunique().rename("raters").reset_index()
    raters = raters[raters["raters"] >= min_raters]
    pool = catalogue[["tmdb_id"]].merge(raters, on="tmdb_id")
    pool = pool.sort_values("tmdb_id", ignore_index=True)
    pool.attrs["rated_elsewhere"] = int((~raters["tmdb_id"].isin(catalogue["tmdb_id"])).sum())
    return pool


def pool_summary(
    pool: pd.DataFrame, catalogue: pd.DataFrame, movielens_rated: set[int]
) -> dict[str, float]:
    """The numbers that say whether the pool is the right shape.

    ``movielens_cold`` is the share of the pool MovieLens never rated: the films
    `StratifiedRanker` currently treats as crowd-less, whose share sets its
    fairness quota. ``movielens_rated`` must hold films with at least one
    MovieLens rating, not every film in ``links.csv`` (see
    `movielens.rated_tmdb_ids`).
    """
    films = catalogue[["tmdb_id", "vote_count"]].merge(pool[["tmdb_id"]], on="tmdb_id")
    votes = pd.to_numeric(films["vote_count"], errors="coerce").fillna(0)
    return {
        "films": len(films),
        "movielens_cold": float((~films["tmdb_id"].isin(movielens_rated)).mean()),
        "median_tmdb_votes": float(votes.median()),
        "zero_tmdb_votes": int((votes == 0).sum()),
        "under_20_tmdb_votes": float((votes < 20).mean()),
    }


def pool_by_year(pool: pd.DataFrame, catalogue: pd.DataFrame, years: range) -> pd.DataFrame:
    """Pool films per release year, against the catalogue: where the horizon bites.

    Uses TMDb's current release year rather than the dump's, which is frozen at
    scrape time and still says 2022 for films that came out in 2024.
    """
    films = catalogue[["tmdb_id", "year"]]
    films = films[films["year"].isin(list(years))]
    in_pool = films["tmdb_id"].isin(pool["tmdb_id"])
    report = (
        pd.DataFrame({"year": films["year"].astype(int), "in_pool": in_pool})
        .groupby("year")["in_pool"]
        .agg(catalogue="size", pool="sum")
        .reindex(list(years), fill_value=0)
        .rename_axis("year")
        .reset_index()
    )
    report["share"] = (report["pool"] / report["catalogue"].replace(0, np.nan)).map(
        lambda v: "" if pd.isna(v) else f"{v:.1%}"
    )
    return report
