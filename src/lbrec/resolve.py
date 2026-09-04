"""Resolve Letterboxd films to TMDb IDs.

The export gives a boxd.it slug, a title and a year -- no external ID. Every
dataset this project joins against (MovieLens ``links.csv``, the IMDb dumps,
TMDb itself) keys on TMDb or IMDb IDs, so this mapping is the join key the whole
pipeline rests on.

Matching is deliberately conservative. A wrong ID is worse than a missing one:
a missing film is visibly absent from the catalog, while a wrong one silently
attaches someone else's synopsis, cast and popularity to a rating. Anything the
rules will not accept goes to ``artifacts/review/unresolved.csv`` for a human
instead of being guessed at.

Manual corrections live in ``overrides/film_id_overrides.csv``, which is
committed, authoritative, and never rewritten by the resolver. Re-running the
resolution pass therefore never loses hand-checked work.

This step emits ``tmdb_id`` only. IMDb IDs and full metadata come from the later
enrichment pass, which fetches ``/movie/{id}`` once per resolved film.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from lbrec.letterboxd import normalize_title
from lbrec.tmdb import TmdbClient

#: Confidence tiers, most to least trusted.
EXACT = "exact"
HIGH = "high"
MEDIUM = "medium"
UNRESOLVED = "unresolved"
OVERRIDE = "override"

#: Acceptance thresholds. Tuned to prefer a miss over a mismatch.
HIGH_TITLE_SCORE = 90.0
MEDIUM_TITLE_SCORE = 85.0
HIGH_YEAR_DELTA = 1
MEDIUM_YEAR_DELTA = 2

#: TMDb keeps film and television in separate namespaces.
MEDIA_MOVIE = "movie"
MEDIA_TV = "tv"

FILM_MAP_COLUMNS = [
    "film_key",
    "title",
    "year",
    "tmdb_id",
    "media_type",
    "tmdb_title",
    "tmdb_year",
    "confidence",
    "title_score",
    "year_delta",
    "source",
]

#: Tiers downstream code may use without a human having looked.
TRUSTED_CONFIDENCES = frozenset({EXACT, HIGH, OVERRIDE})

#: Tiers that go to the review file. MEDIUM is included deliberately: it is
#: right most of the time, and confirming a handful of rows by hand is cheaper
#: than one silently wrong ID propagating through every later stage.
REVIEW_CONFIDENCES = frozenset({UNRESOLVED, MEDIUM})

REVIEW_COLUMNS = [
    "film_key",
    "title",
    "year",
    "letterboxd_url",
    "tmdb_id",
    "media_type",
    "auto_match",
    "suggestion_1",
    "suggestion_2",
    "suggestion_3",
    "note",
]


#: Confidence ordering, for comparing two candidate matches.
_RANK = {EXACT: 3, HIGH: 2, MEDIUM: 1, UNRESOLVED: 0}

#: Leading/trailing articles that routinely differ between Letterboxd and TMDb
#: ("The Fisher King" vs "Fisher King", "Fisher King, The"). Italian "i" and
#: "gli" are omitted deliberately: "i" collides with the English pronoun, and
#: the gain does not justify mangling titles like "I Am Cuba".
_ARTICLES = frozenset(
    """
    the a an
    le la les un une l
    el los las una
    il lo uno
    der die das ein eine
    o os as um uma
    """.split()
)


def title_variants(normalized: str) -> set[str]:
    """Forms of a normalised title to compare against.

    Variants can only raise the similarity score, never lower it, so this
    widens recall without weakening the year check that guards precision.
    """
    variants = {normalized}
    tokens = normalized.split()
    if len(tokens) > 1:
        if tokens[0] in _ARTICLES:
            variants.add(" ".join(tokens[1:]))
        if tokens[-1] in _ARTICLES:  # TMDb's "Fisher King, The" form
            variants.add(" ".join(tokens[:-1]))
    return variants


#: Roman numerals used as sequel markers. Deliberately small: only the forms
#: that actually appear at the end of film titles.
_ROMAN = frozenset("i ii iii iv v vi vii viii ix x xi xii xiii".split())


def sequel_marker(normalized: str) -> str | None:
    """The trailing sequel designator of a title, if it has one.

    Sequel numbers are one or two characters inside otherwise identical titles,
    so fuzzy similarity barely notices them: "drunken master ii" scores 97
    against "drunken master iii". Comparing the markers explicitly is the only
    reliable way to keep entries in a series apart.

    Returns ``None`` for single-token titles, so a film actually called "X" or
    "Ran" is not read as a sequel marker.
    """
    tokens = normalized.split()
    if len(tokens) < 2:
        return None
    last = tokens[-1]
    if last.isdigit() or last in _ROMAN:
        return last
    return None


def sequel_conflict(query: str, candidate_forms: set[str]) -> bool:
    """True when no candidate title carries the same sequel marker as the query.

    Also catches the asymmetric case ("Blade Runner" vs "Blade Runner 2049"),
    where one title has a trailing number and the other does not.
    """
    return all(sequel_marker(query) != sequel_marker(form) for form in candidate_forms)


def letterboxd_url(film_key: str) -> str:
    """Clickable link for the human review pass."""
    return "" if film_key.startswith("ty:") else f"https://boxd.it/{film_key}"


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------
OVERRIDE_COLUMNS = ["film_key", "tmdb_id", "media_type", "title", "note"]


@dataclass(frozen=True)
class Override:
    """One hand-written correction.

    ``tmdb_id is None`` means "deliberately unmatchable, stop asking", which is
    distinct from a film that was simply never reviewed.
    """

    tmdb_id: int | None
    media_type: str = MEDIA_MOVIE


def load_overrides(path: Path) -> dict[str, Override]:
    """Read hand-written corrections.

    ``media_type`` matters: a series resolved by hand must keep its TMDb
    namespace, or it would later be looked up as a film and silently vanish.

    The file is append-only, so a later row for the same ``film_key`` wins. That
    is how a correction is recorded without rewriting history.
    """
    if not path.exists():
        return {}
    overrides: dict[str, Override] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("film_key") or "").strip()
            if not key:
                continue
            raw = (row.get("tmdb_id") or "").strip()
            media = (row.get("media_type") or "").strip().lower() or MEDIA_MOVIE
            try:
                tmdb_id = int(raw) if raw else None
            except ValueError:
                tmdb_id = None
            overrides[key] = Override(
                tmdb_id=tmdb_id,
                media_type=media if media in {MEDIA_MOVIE, MEDIA_TV} else MEDIA_MOVIE,
            )
    return overrides


def append_overrides(path: Path, rows: list[dict[str, str]]) -> int:
    """Append reviewed rows. Never rewrites existing lines."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERRIDE_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OVERRIDE_COLUMNS})
    return len(rows)


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Match:
    tmdb_id: int | None
    tmdb_title: str
    tmdb_year: int | None
    confidence: str
    title_score: float
    year_delta: int | None
    media_type: str = MEDIA_MOVIE


NO_MATCH = Match(None, "", None, UNRESOLVED, 0.0, None)


def tv_as_movie(result: dict) -> dict:
    """Reshape a TV search result into the movie field names.

    Lets one scoring path serve both namespaces instead of duplicating the
    thresholds, the sequel guard and the article variants for television.
    """
    return {
        "id": result.get("id"),
        "title": result.get("name") or "",
        "original_title": result.get("original_name") or "",
        "release_date": result.get("first_air_date") or "",
        "popularity": result.get("popularity") or 0.0,
    }


def _candidate_year(candidate: dict) -> int | None:
    date = candidate.get("release_date") or ""
    try:
        return int(date[:4])
    except (TypeError, ValueError):
        return None


def candidate_titles(candidate: dict) -> set[str]:
    """Every normalised title form TMDb offers for a candidate."""
    forms: set[str] = set()
    for key in ("title", "original_title"):
        normalized = normalize_title(candidate.get(key) or "")
        if normalized:
            forms |= title_variants(normalized)
    return forms


def title_score(query: str, candidate: dict) -> float:
    """Best similarity across TMDb's localised and original titles.

    A film exported under its English title may sit under its original title on
    TMDb and vice versa, so both are compared -- along with article-stripped
    variants of each -- and the best score wins.
    """
    targets = title_variants(normalize_title(query))
    scores = [
        max(fuzz.ratio(target, value), fuzz.token_sort_ratio(target, value))
        for target in targets
        for value in candidate_titles(candidate)
    ]
    return max(scores, default=0.0)


def score_candidates(title: str, year: int | None, candidates: list[dict]) -> Match:
    """Pick the best acceptable candidate, or an unresolved Match.

    Candidates are compared on (confidence tier, literal title match, title
    score). The literal term matters because article stripping makes "The
    Stalker" an exact-tier match for "Stalker": a title that matches verbatim
    must outrank one that only matches after a variant was applied.

    Remaining ties are broken by TMDb popularity. That is a popularity bias, but
    an appropriate one: this is *identity* resolution, and for a genuinely
    ambiguous title+year the better-known film is the one Letterboxd means. It
    has no bearing on ranking, which is debiased separately.
    """
    best: Match | None = None
    best_key = (0, 0, 0.0)
    normalized = normalize_title(title)
    query_forms = title_variants(normalized)

    ranked = sorted(candidates, key=lambda c: float(c.get("popularity") or 0.0), reverse=True)
    for candidate in ranked:
        tmdb_id = candidate.get("id")
        if tmdb_id is None:
            continue
        score = title_score(title, candidate)
        candidate_year = _candidate_year(candidate)
        delta = None if (year is None or candidate_year is None) else abs(candidate_year - year)

        forms = candidate_titles(candidate)
        # A differing sequel number means a different film, whatever the
        # similarity score says. Reject rather than demote: entries in a series
        # are exactly where a confident wrong ID does the most damage.
        if sequel_conflict(normalized, forms):
            continue

        literal_title = normalized in {
            normalize_title(candidate.get(key) or "") for key in ("title", "original_title")
        }
        exact_title = bool(query_forms & forms)
        if exact_title and delta is not None and delta <= HIGH_YEAR_DELTA:
            confidence = EXACT
        elif score >= HIGH_TITLE_SCORE and delta is not None and delta <= HIGH_YEAR_DELTA:
            confidence = HIGH
        elif score >= MEDIUM_TITLE_SCORE and delta is not None and delta <= MEDIUM_YEAR_DELTA:
            confidence = MEDIUM
        else:
            continue

        match = Match(
            tmdb_id=int(tmdb_id),
            tmdb_title=candidate.get("title") or candidate.get("original_title") or "",
            tmdb_year=candidate_year,
            confidence=confidence,
            title_score=float(score),
            year_delta=delta,
        )
        # Candidates arrive popularity-ordered, so a strict improvement is
        # required to displace an earlier match and popularity breaks ties.
        key = (_RANK[confidence], int(literal_title), score)
        if key > best_key:
            best, best_key = match, key
        # Nothing can beat a verbatim title on a matching year.
        if confidence == EXACT and literal_title:
            break

    return best or NO_MATCH


#: Confidence ordering, for comparing two candidate matches.
_RANK = {EXACT: 3, HIGH: 2, MEDIUM: 1, UNRESOLVED: 0}


def resolve_film(
    client: TmdbClient, title: str, year: int | None, *, allow_tv: bool = True
) -> Match:
    """Search TMDb for one entry: film first, then television.

    Searches run lazily and stop at the first exact hit, so the common case
    costs a single request. TMDb's ``primary_release_year`` filter can hide the
    right film when release dates disagree across regions, which is what the
    unconstrained retry is for.

    Television is only consulted once the film namespace has failed to produce
    an exact match, so a film is never displaced by a like-named series. Matches
    are stamped with ``media_type``: TV entries are taste signal only and are
    excluded from the recommendation catalog.
    """
    if not title.strip():
        return NO_MATCH

    best = NO_MATCH

    movie_queries: list[dict] = []
    if year is not None:
        movie_queries.append({"primary_release_year": year})
    movie_queries.append({})
    for params in movie_queries:
        match = score_candidates(title, year, client.search_movie(title, **params))
        if match.confidence == EXACT:
            return match
        if _RANK[match.confidence] > _RANK[best.confidence]:
            best = match

    if not allow_tv:
        return best

    tv_queries: list[dict] = []
    if year is not None:
        tv_queries.append({"first_air_date_year": year})
    tv_queries.append({})
    for params in tv_queries:
        results = [tv_as_movie(result) for result in client.search_tv(title, **params)]
        match = score_candidates(title, year, results)
        if match.confidence == UNRESOLVED:
            continue
        match = replace(match, media_type=MEDIA_TV)
        if match.confidence == EXACT:
            return match
        # A film match of equal confidence wins: only improve on `best`.
        if _RANK[match.confidence] > _RANK[best.confidence]:
            best = match
    return best


# --------------------------------------------------------------------------
# batch resolution
# --------------------------------------------------------------------------
def resolve_films(
    client: TmdbClient,
    films: pd.DataFrame,
    overrides: dict[str, Override],
    *,
    progress=None,
) -> pd.DataFrame:
    """Resolve every film, honouring overrides without touching the API."""
    records = []
    for film in films.itertuples(index=False):
        year = int(film.year) if pd.notna(film.year) else None

        if film.film_key in overrides:
            override = overrides[film.film_key]
            tmdb_id = override.tmdb_id
            records.append(
                {
                    "film_key": film.film_key,
                    "title": film.title,
                    "year": year,
                    "tmdb_id": tmdb_id,
                    "media_type": override.media_type if tmdb_id is not None else None,
                    "tmdb_title": "",
                    "tmdb_year": None,
                    "confidence": OVERRIDE if tmdb_id is not None else UNRESOLVED,
                    "title_score": None,
                    "year_delta": None,
                    "source": OVERRIDE,
                }
            )
        else:
            match = resolve_film(client, film.title, year)
            records.append(
                {
                    "film_key": film.film_key,
                    "title": film.title,
                    "year": year,
                    "tmdb_id": match.tmdb_id,
                    "media_type": match.media_type if match.tmdb_id is not None else None,
                    "tmdb_title": match.tmdb_title,
                    "tmdb_year": match.tmdb_year,
                    "confidence": match.confidence,
                    "title_score": match.title_score,
                    "year_delta": match.year_delta,
                    "source": "auto",
                }
            )
        if progress is not None:
            progress.update(1)

    frame = pd.DataFrame.from_records(records, columns=FILM_MAP_COLUMNS)
    for column in ("year", "tmdb_id", "tmdb_year", "year_delta"):
        frame[column] = frame[column].astype("Int64")
    frame["title_score"] = frame["title_score"].astype("Float64")
    return frame


def build_review_table(client: TmdbClient, film_map: pd.DataFrame) -> pd.DataFrame:
    """Rows a human must decide on, with TMDb's top guesses for context.

    Covers unresolved films and medium-confidence ones. ``auto_match`` shows
    what the matcher proposed, if anything; ``tmdb_id`` is left blank so a row
    only counts as reviewed once a person has filled it in.

    Rows already decided by hand are excluded, including deliberate non-matches:
    once someone has recorded that a film is not on TMDb, asking again every run
    would bury the rows that still need attention.
    """
    pending = film_map[
        film_map["confidence"].isin(REVIEW_CONFIDENCES) & (film_map["source"] != OVERRIDE)
    ]
    records = []
    for film in pending.itertuples(index=False):
        year = int(film.year) if pd.notna(film.year) else None
        candidates = client.search_movie(film.title)[:3]
        suggestions = [
            f"{c.get('id')} | {c.get('title')} ({(c.get('release_date') or '????')[:4]})"
            for c in candidates
        ]
        suggestions += [""] * (3 - len(suggestions))
        auto = (
            f"{film.tmdb_id} | {film.tmdb_title} ({film.tmdb_year}) "
            f"[{film.confidence}, {film.media_type}]"
            if pd.notna(film.tmdb_id)
            else ""
        )
        records.append(
            {
                "film_key": film.film_key,
                "title": film.title,
                "year": year,
                "letterboxd_url": letterboxd_url(film.film_key),
                "tmdb_id": "",
                "media_type": MEDIA_MOVIE,
                "auto_match": auto,
                "suggestion_1": suggestions[0],
                "suggestion_2": suggestions[1],
                "suggestion_3": suggestions[2],
                "note": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=REVIEW_COLUMNS)


def duplicate_ids(film_map: pd.DataFrame) -> pd.DataFrame:
    """Films sharing one TMDb ID.

    Usually a real mistake and it would double-count that film's interactions. Occasionally
    legitimate, when Letterboxd splits something TMDb keeps whole, so this is
    surfaced rather than enforced.
    """
    resolved = film_map[film_map["tmdb_id"].notna()]
    duplicates = resolved[resolved.duplicated("tmdb_id", keep=False)]
    return duplicates.sort_values(["tmdb_id", "film_key"], ignore_index=True)


def read_reviewed(path: Path) -> list[dict[str, str]]:
    """Collect rows a human filled in, ready to append to the overrides file."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("film_key") or "").strip()
            tmdb_id = (row.get("tmdb_id") or "").strip()
            note = (row.get("note") or "").strip()
            # A blank tmdb_id with no note means "not reviewed yet", so skip it.
            # A note alone records a deliberate decision not to match.
            if not key or (not tmdb_id and not note):
                continue
            media = (row.get("media_type") or "").strip().lower()
            rows.append(
                {
                    "film_key": key,
                    "tmdb_id": tmdb_id,
                    "media_type": media if media in {MEDIA_MOVIE, MEDIA_TV} else MEDIA_MOVIE,
                    "title": (row.get("title") or "").strip(),
                    "note": note,
                }
            )
    return rows
