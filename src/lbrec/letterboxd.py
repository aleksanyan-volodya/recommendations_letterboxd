"""Parse a Letterboxd export directory into two tidy tables.

The export is seven-plus CSVs with overlapping semantics :

1. ``diary.csv`` stores the **diary entry** URI (a 6-char slug), not the film
   URI. Its slugs share no values with ``watched.csv``. Diary rows must be
   joined on ``(title, year)`` instead.
2. List files are three blocks in one CSV: a magic line, a list-metadata
   header + row, a blank line, then the film header + rows.
3. Every ``Date`` column is a **logging** date, not a viewing date. Films
   backfilled when the account was created all share one timestamp. The only
   true viewing chronology is ``diary.Watched Date``, which most users do not keep.

Signal reliability, for whoever builds features on top of this:

===============  ==========  ==================================================
kind             coverage    notes
===============  ==========  ==================================================
rating           high        the primary label; full 1-5 scale
watched          high        superset of rating and like
watchlist        high        positive intent, but availability-biased
like             medium      binary, sits on top of a rating
list_entry       low         idiosyncratic, membership only
diary            **low**     optional enrichment. Sparse here and absent for
                             most users. Carries the only real watch dates and
                             the rewatch flag. **Nothing downstream may require
                             it** -- treat every diary field as possibly empty.
===============  ==========  ==================================================

Outputs
-------
films_local
    One row per distinct film seen anywhere in the export.
interactions
    Long format, one row per (film, interaction). Every signal the export
    carries lands here with the same schema so downstream code has a single
    door. Ingesting a signal is not weighting it; that is a modelling choice
    made later, against the coverage report.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LIST_MAGIC = "Letterboxd list export"

#: Interaction kinds emitted into the ``kind`` column.
KIND_RATING = "rating"
KIND_WATCHED = "watched"
KIND_WATCHLIST = "watchlist"
KIND_LIKE = "like"
KIND_DIARY = "diary"
KIND_LIST = "list_entry"

#: Kinds that are sparse or user-dependent. Nothing may require these.
OPTIONAL_KINDS = frozenset({KIND_DIARY, KIND_LIST})

INTERACTION_COLUMNS = [
    "film_key",
    "title",
    "year",
    "kind",
    "value",
    "event_date",
    "watched_date",
    "rewatch",
    "list_name",
    "tags",
    "source",
    "deleted",
]


# --------------------------------------------------------------------------
# low-level readers
# --------------------------------------------------------------------------
def _open_csv(path: Path):
    # utf-8-sig strips a BOM if present; newline="" lets csv handle CRLF itself.
    return path.open("r", newline="", encoding="utf-8-sig")


def read_simple_csv(path: Path) -> list[dict[str, str]]:
    """Read a flat Letterboxd CSV (ratings, watched, watchlist, diary, likes)."""
    if not path.exists():
        return []
    with _open_csv(path) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@dataclass(frozen=True)
class ListFile:
    """A parsed Letterboxd list export."""

    name: str
    tags: str
    date: str
    entries: list[dict[str, str]]


def read_list_csv(path: Path) -> ListFile | None:
    """Parse the three-block list export format.

    Returns ``None`` for files that do not carry the list magic line, so a
    stray CSV in ``lists/`` cannot be silently misread as a list.
    """
    if not path.exists():
        return None
    with _open_csv(path) as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0] or not rows[0][0].startswith(LIST_MAGIC):
        return None

    # Block 1: list metadata (header then one row), ending at the blank line.
    blank = next((i for i, row in enumerate(rows) if not any(cell.strip() for cell in row)), None)
    if blank is None or blank < 3:
        return None
    meta = dict(zip(rows[1], rows[2], strict=False))

    # Block 2: the films.
    film_rows = [row for row in rows[blank + 1 :] if any(cell.strip() for cell in row)]
    if film_rows:
        header, *body = film_rows
        entries = [dict(zip(header, row, strict=False)) for row in body]
    else:
        entries = []

    return ListFile(
        name=meta.get("Name", path.stem),
        tags=meta.get("Tags", ""),
        date=meta.get("Date", ""),
        entries=entries,
    )


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Casefold and strip accents and punctuation, for fallback matching only.

    Never used as a display value, and never the sole basis for accepting an
    external ID match and the year has to agree too.
    """
    decomposed = unicodedata.normalize("NFKD", title or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACE.sub(" ", _PUNCT.sub(" ", stripped)).strip().casefold()


def film_slug(uri: str) -> str | None:
    """Extract the boxd.it slug from a Letterboxd URI."""
    if not uri:
        return None
    slug = uri.rstrip("/").rsplit("/", 1)[-1].strip()
    return slug or None


def _year(raw: str | None) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def title_year_key(title: str, year: int | None) -> str:
    """Synthetic film key for rows carrying no film URI (e.g. deleted diary entries)."""
    return f"ty:{normalize_title(title)}:{year if year is not None else '?'}"


# --------------------------------------------------------------------------
# export loading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Export:
    films_local: pd.DataFrame
    interactions: pd.DataFrame


def _uri_keyed_rows(
    rows: list[dict[str, str]],
    *,
    kind: str,
    source: str,
    deleted: bool,
    value_column: str | None = None,
) -> list[dict]:
    """Build interaction records for files keyed by the film URI."""
    records = []
    for row in rows:
        key = film_slug(row.get("Letterboxd URI", ""))
        if key is None:
            continue
        value = None
        if value_column:
            try:
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError):
                value = None
        records.append(
            {
                "film_key": key,
                "title": row.get("Name", ""),
                "year": _year(row.get("Year")),
                "kind": kind,
                "value": value,
                "event_date": row.get("Date") or None,
                "watched_date": None,
                "rewatch": None,
                "list_name": None,
                "tags": row.get("Tags") or None,
                "source": source,
                "deleted": deleted,
            }
        )
    return records


def _diary_rows(
    rows: list[dict[str, str]],
    *,
    resolver: dict[tuple[str, int | None], str],
    source: str,
    deleted: bool,
) -> list[dict]:
    """Build diary interactions, joined on (title, year) since the URI is an entry URI."""
    records = []
    for row in rows:
        title = row.get("Name", "")
        year = _year(row.get("Year"))
        key = resolver.get((normalize_title(title), year)) or title_year_key(title, year)
        try:
            value = float(row["Rating"])
        except (KeyError, TypeError, ValueError):
            value = None
        records.append(
            {
                "film_key": key,
                "title": title,
                "year": year,
                "kind": KIND_DIARY,
                "value": value,
                "event_date": row.get("Date") or None,
                "watched_date": row.get("Watched Date") or None,
                "rewatch": bool((row.get("Rewatch") or "").strip()),
                "list_name": None,
                "tags": row.get("Tags") or None,
                "source": source,
                "deleted": deleted,
            }
        )
    return records


def _list_rows(list_file: ListFile, *, source: str, deleted: bool) -> list[dict]:
    records = []
    for entry in list_file.entries:
        title = entry.get("Name", "")
        year = _year(entry.get("Year"))
        key = film_slug(entry.get("URL", "")) or title_year_key(title, year)
        records.append(
            {
                "film_key": key,
                "title": title,
                "year": year,
                "kind": KIND_LIST,
                "value": None,
                "event_date": list_file.date or None,
                "watched_date": None,
                "rewatch": None,
                "list_name": list_file.name,
                "tags": list_file.tags or None,
                "source": source,
                "deleted": deleted,
            }
        )
    return records


def load_export(export_dir: Path, *, include_deleted: bool = True) -> Export:
    """Parse an export directory into ``films_local`` and ``interactions``.

    ``deleted/`` holds entries whose Letterboxd film page was removed or merged
    (usually duplicate TMDb records). Those are still real watches, so they are
    included by default and flagged with ``deleted=True`` rather than dropped.
    ``orphaned/`` has the same shape and is treated identically.

    Missing files are normal, not an error: most users have no reviews, no
    lists, and an empty or absent diary.
    """
    export_dir = Path(export_dir)
    records: list[dict] = []

    # Pass 1: the URI-keyed files. These also seed the (title, year) -> key
    # resolver that the diary files need.
    uri_specs = [
        ("ratings.csv", KIND_RATING, "Rating"),
        ("watched.csv", KIND_WATCHED, None),
        ("watchlist.csv", KIND_WATCHLIST, None),
        ("likes/films.csv", KIND_LIKE, None),
    ]
    for relative, kind, value_column in uri_specs:
        rows = read_simple_csv(export_dir / relative)
        records.extend(
            _uri_keyed_rows(
                rows, kind=kind, source=relative, deleted=False, value_column=value_column
            )
        )

    resolver: dict[tuple[str, int | None], str] = {}
    for record in records:
        resolver.setdefault((normalize_title(record["title"]), record["year"]), record["film_key"])

    # Pass 2: lists (URI-keyed, but nested and in the three-block format).
    for directory, deleted in (("lists", False), ("deleted/lists", True)):
        if deleted and not include_deleted:
            continue
        for path in sorted((export_dir / directory).glob("*.csv")):
            list_file = read_list_csv(path)
            if list_file is None:
                continue
            records.extend(
                _list_rows(list_file, source=f"{directory}/{path.name}", deleted=deleted)
            )

    # Pass 3: diary files, resolved by (title, year). Optional, often empty.
    diary_specs = [("diary.csv", False), ("deleted/diary.csv", True), ("orphaned/diary.csv", True)]
    for relative, deleted in diary_specs:
        if deleted and not include_deleted:
            continue
        rows = read_simple_csv(export_dir / relative)
        records.extend(_diary_rows(rows, resolver=resolver, source=relative, deleted=deleted))

    interactions = pd.DataFrame.from_records(records, columns=INTERACTION_COLUMNS)
    interactions["year"] = interactions["year"].astype("Int64")
    interactions["value"] = interactions["value"].astype("Float64")
    interactions["rewatch"] = interactions["rewatch"].astype("boolean")
    interactions["deleted"] = interactions["deleted"].astype("boolean")
    # Logging dates
    for column in ("event_date", "watched_date"):
        interactions[column] = pd.to_datetime(interactions[column], errors="coerce")

    return Export(films_local=build_films_local(interactions), interactions=interactions)


def build_films_local(interactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse interactions into one row per distinct film.

    Title and year are taken from the most frequent spelling across sources, so
    one odd row in one file cannot rename the film.
    """
    columns = [
        "film_key",
        "title",
        "year",
        "n_interactions",
        "kinds",
        "synthetic_key",
        "deleted_only",
    ]
    if interactions.empty:
        return pd.DataFrame(columns=columns)

    def _mode(series: pd.Series):
        modes = series.mode(dropna=True)
        return modes.iloc[0] if len(modes) else pd.NA

    films = (
        interactions.groupby("film_key", as_index=False)
        .agg(
            title=("title", _mode),
            year=("year", _mode),
            n_interactions=("kind", "size"),
            kinds=("kind", lambda s: "|".join(sorted(set(s)))),
            deleted_only=("deleted", "all"),
        )
        .sort_values("film_key", ignore_index=True)
    )
    films["year"] = films["year"].astype("Int64")
    films["synthetic_key"] = films["film_key"].str.startswith("ty:")
    films["deleted_only"] = films["deleted_only"].astype("boolean")
    return films[columns]


#: Kinds that prove the film was actually seen.
SEEN_KINDS = frozenset({KIND_WATCHED, KIND_RATING, KIND_DIARY})


def film_status(interactions: pd.DataFrame) -> pd.DataFrame:
    """One authoritative row per film: seen, rating, liked, watchlist.

    ``seen`` is the union of watched, rated and diarised rather than just
    ``watched.csv``, because a deleted Letterboxd film entry drops the watch
    record while leaving the diary and watchlist entries in place.

    ``watchlist_pending`` is the watchlist minus seen films and is
    the set to use both for recommending and as held-out positives.
    """
    columns = [
        "film_key",
        "title",
        "year",
        "seen",
        "rating",
        "liked",
        "on_watchlist",
        "watchlist_pending",
    ]
    if interactions.empty:
        return pd.DataFrame(columns=columns)

    films = build_films_local(interactions)[["film_key", "title", "year"]].copy()
    kind = interactions["kind"]

    def _keys(mask: pd.Series) -> set[str]:
        return set(interactions.loc[mask, "film_key"])

    # ratings.csv holds one row per film; diary ratings only fill gaps.
    ratings = (
        interactions[kind == KIND_RATING].drop_duplicates("film_key").set_index("film_key")["value"]
    )
    diary_ratings = (
        interactions[(kind == KIND_DIARY) & interactions["value"].notna()]
        .sort_values("watched_date")
        .drop_duplicates("film_key", keep="last")
        .set_index("film_key")["value"]
    )

    films["seen"] = films["film_key"].isin(_keys(kind.isin(SEEN_KINDS)))
    films["rating"] = (
        films["film_key"]
        .map(ratings)
        .fillna(films["film_key"].map(diary_ratings))
        .astype("Float64")
    )
    films["liked"] = films["film_key"].isin(_keys(kind == KIND_LIKE))
    films["on_watchlist"] = films["film_key"].isin(_keys(kind == KIND_WATCHLIST))
    films["watchlist_pending"] = films["on_watchlist"] & ~films["seen"]
    return films[columns]


def coverage_report(interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-kind coverage, so sparse signals stay visible instead of being assumed.

    Reported every ingest run. If ``diary`` shows 71 rows against 788 ratings,
    that should be impossible to miss when choosing features later.
    """
    if interactions.empty:
        return pd.DataFrame(columns=["kind", "rows", "films", "with_value", "optional"])

    report = (
        interactions.groupby("kind", as_index=False)
        .agg(
            rows=("film_key", "size"),
            films=("film_key", "nunique"),
            with_value=("value", lambda s: int(s.notna().sum())),
        )
        .sort_values("rows", ascending=False, ignore_index=True)
    )
    report["optional"] = report["kind"].isin(OPTIONAL_KINDS)
    return report
