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
from dataclasses import dataclass
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

FILM_MAP_COLUMNS = [
    "film_key",
    "title",
    "year",
    "tmdb_id",
    "tmdb_title",
    "tmdb_year",
    "confidence",
    "title_score",
    "year_delta",
    "source",
]

REVIEW_COLUMNS = [
    "film_key",
    "title",
    "year",
    "letterboxd_url",
    "tmdb_id",
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


def letterboxd_url(film_key: str) -> str:
    """Clickable link for the human review pass."""
    return "" if film_key.startswith("ty:") else f"https://boxd.it/{film_key}"


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------
def load_overrides(path: Path) -> dict[str, int | None]:
    """Read hand-written corrections.

    An empty ``tmdb_id`` means "deliberately unmatchable, stop asking" and is
    stored as ``None`` -- distinct from a film that was simply never reviewed.
    """
    if not path.exists():
        return {}
    overrides: dict[str, int | None] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("film_key") or "").strip()
            if not key:
                continue
            raw = (row.get("tmdb_id") or "").strip()
            try:
                overrides[key] = int(raw) if raw else None
            except ValueError:
                overrides[key] = None
    return overrides


def append_overrides(path: Path, rows: list[dict[str, str]]) -> int:
    """Append reviewed rows. Never rewrites existing lines."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["film_key", "tmdb_id", "title", "note"])
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
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

        literal_title = normalized in {
            normalize_title(candidate.get(key) or "") for key in ("title", "original_title")
        }
        exact_title = bool(query_forms & candidate_titles(candidate))
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

    return best or Match(None, "", None, UNRESOLVED, 0.0, None)


#: Confidence ordering, for comparing two candidate matches.
_RANK = {EXACT: 3, HIGH: 2, MEDIUM: 1, UNRESOLVED: 0}


def resolve_film(client: TmdbClient, title: str, year: int | None) -> Match:
    """Search TMDb for one film: year-constrained first, then unconstrained.

    Searches run lazily and stop at the first exact hit, so the common case
    costs a single request. TMDb's ``primary_release_year`` filter can hide the
    right film when its release dates disagree across regions, which is what
    the unconstrained retry is for.
    """
    if not title.strip():
        return Match(None, "", None, UNRESOLVED, 0.0, None)

    queries: list[dict] = []
    if year is not None:
        queries.append({"primary_release_year": year})
    queries.append({})

    best = Match(None, "", None, UNRESOLVED, 0.0, None)
    for params in queries:
        match = score_candidates(title, year, client.search_movie(title, **params))
        if match.confidence == EXACT:
            return match
        if _RANK[match.confidence] > _RANK[best.confidence]:
            best = match
    return best


# --------------------------------------------------------------------------
# batch resolution
# --------------------------------------------------------------------------
def resolve_films(
    client: TmdbClient,
    films: pd.DataFrame,
    overrides: dict[str, int | None],
    *,
    progress=None,
) -> pd.DataFrame:
    """Resolve every film, honouring overrides without touching the API."""
    records = []
    for film in films.itertuples(index=False):
        year = int(film.year) if pd.notna(film.year) else None

        if film.film_key in overrides:
            tmdb_id = overrides[film.film_key]
            records.append(
                {
                    "film_key": film.film_key,
                    "title": film.title,
                    "year": year,
                    "tmdb_id": tmdb_id,
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

    ``tmdb_id`` is left blank for the reviewer to fill in.
    """
    unresolved = film_map[film_map["confidence"] == UNRESOLVED]
    records = []
    for film in unresolved.itertuples(index=False):
        year = int(film.year) if pd.notna(film.year) else None
        candidates = client.search_movie(film.title)[:3]
        suggestions = [
            f"{c.get('id')} | {c.get('title')} ({(c.get('release_date') or '????')[:4]})"
            for c in candidates
        ]
        suggestions += [""] * (3 - len(suggestions))
        records.append(
            {
                "film_key": film.film_key,
                "title": film.title,
                "year": year,
                "letterboxd_url": letterboxd_url(film.film_key),
                "tmdb_id": "",
                "suggestion_1": suggestions[0],
                "suggestion_2": suggestions[1],
                "suggestion_3": suggestions[2],
                "note": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=REVIEW_COLUMNS)


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
            rows.append(
                {
                    "film_key": key,
                    "tmdb_id": tmdb_id,
                    "title": (row.get("title") or "").strip(),
                    "note": note,
                }
            )
    return rows
