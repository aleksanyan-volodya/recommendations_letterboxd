"""Project paths and settings.

Everything derived lives under ``artifacts/`` and is reproducible from the raw
Letterboxd export plus the external sources.

**Nothing is keyed to one particular person.** Artifacts split in two:

*Per user* -- what a given export says about that person: their films, their
interactions, their status table. These live under ``artifacts/users/<user>/``
so ingesting a friend's export cannot overwrite anyone else's.

*Shared* -- facts about films rather than about people: the Letterboxd-to-TMDb
mapping, TMDb metadata, the manual ID overrides, MovieLens. A film's identity
does not depend on who watched it, so resolving it once serves everybody and the
HTTP cache and override file are reused across users.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default user when none is named. A label for one export, not an identity.
DEFAULT_USER = "me"

_SAFE_USER = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def normalize_user(user: str) -> str:
    """Validate a user label, since it becomes a directory name.

    Rejects anything that could escape the artifacts tree or collide by case.
    """
    slug = (user or "").strip().lower()
    if not _SAFE_USER.fullmatch(slug):
        raise ValueError(
            f"invalid user label {user!r}: use lower-case letters, digits, '-' or '_' "
            "(max 64 characters)"
        )
    return slug


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class UserPaths:
    """Where one person's derived tables live."""

    user: str
    root: Path

    @property
    def films_local(self) -> Path:
        return self.root / "films_local.parquet"

    @property
    def interactions(self) -> Path:
        return self.root / "interactions.parquet"

    @property
    def film_status(self) -> Path:
        """One row per film: seen, rating, liked, watchlist_pending."""
        return self.root / "film_status.parquet"

    def ensure(self) -> UserPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one run."""

    export_dir: Path
    artifacts_dir: Path
    tmdb_token: str | None
    tmdb_api_key: str | None
    tmdb_rps: float
    default_user: str = DEFAULT_USER

    # --- derived locations -------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        """Raw HTTP responses, keyed by request. Makes re-runs free and reproducible."""
        return self.artifacts_dir / "cache"

    @property
    def external_dir(self) -> Path:
        """Third-party bulk datasets (MovieLens, IMDb) as downloaded."""
        return self.artifacts_dir / "external"

    @property
    def processed_dir(self) -> Path:
        """Tables this project produces and later stages consume."""
        return self.artifacts_dir / "processed"

    @property
    def review_dir(self) -> Path:
        """CSVs a human is expected to open and edit."""
        return self.artifacts_dir / "review"

    @property
    def users_dir(self) -> Path:
        """One subdirectory per ingested export."""
        return self.artifacts_dir / "users"

    def user(self, user: str | None = None) -> UserPaths:
        """Paths for one person's derived tables."""
        slug = normalize_user(user or self.default_user)
        return UserPaths(user=slug, root=self.users_dir / slug)

    def known_users(self) -> list[str]:
        """Every export ingested so far.

        Resolution runs over the union of these, so one pass covers everybody and
        the TMDb cache and overrides are shared rather than repeated per user.
        """
        if not self.users_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.users_dir.iterdir()
            if path.is_dir() and (path / "films_local.parquet").exists()
        )

    @property
    def film_map_path(self) -> Path:
        """Shared: film identity does not depend on who watched it."""
        return self.processed_dir / "film_map.parquet"

    @property
    def movielens_zip(self) -> Path:
        """The downloaded MovieLens archive, as-is."""
        return self.external_dir / "ml-32m.zip"

    @property
    def movielens_dir(self) -> Path:
        """Parquet conversions of the archive members."""
        return self.external_dir / "movielens"

    @property
    def item_factors_path(self) -> Path:
        """MovieLens latent item vectors. Shared: learned from other people only."""
        return self.processed_dir / "movielens_item_factors.parquet"

    @property
    def content_vectors_path(self) -> Path:
        """Dense content vectors for MovieLens items, keyed by movieId.

        Cached because building them means vectorising 84k films, which costs
        minutes; nothing about them depends on ratings, so one build serves every
        experiment and every user.
        """
        return self.processed_dir / "movielens_content_vectors.parquet"

    @property
    def samlearner_zip(self) -> Path:
        """Letterboxd crowd dump, scraped Nov 2020 -> Mar 2022. Carries TMDb ids."""
        return self.external_dir / "letterboxd-samlearner.zip"

    @property
    def freeth_zip(self) -> Path:
        """Letterboxd crowd dump, scraped Oct 2023. Slugs only, no TMDb ids."""
        return self.external_dir / "letterboxd-film-ratings.zip"

    @property
    def crowd_ratings_path(self) -> Path:
        """Both dumps merged, one row per (member, film), keyed by TMDb id."""
        return self.processed_dir / "crowd_ratings.parquet"

    @property
    def crowd_films_path(self) -> Path:
        """Every Letterboxd slug the dumps mention, and how it got its TMDb id."""
        return self.processed_dir / "crowd_films.parquet"

    @property
    def pool_path(self) -> Path:
        """The candidate pool: recommendable films with a Letterboxd audience."""
        return self.processed_dir / "pool.parquet"

    @property
    def tmdb_export_path(self) -> Path:
        """TMDb's daily id dump, as downloaded."""
        return self.external_dir / "tmdb_movie_ids.json.gz"

    @property
    def tmdb_store_dir(self) -> Path:
        """Sharded bulk enrichment of the whole of TMDb. Shared across users."""
        return self.external_dir / "tmdb_films"

    @property
    def catalogue_path(self) -> Path:
        """Every film we are allowed to recommend. Shared across users."""
        return self.processed_dir / "catalogue.parquet"

    @property
    def films_tmdb_path(self) -> Path:
        """TMDb metadata, one row per resolved title."""
        return self.processed_dir / "films_tmdb.parquet"

    @property
    def unresolved_path(self) -> Path:
        """Written for the human review pass."""
        return self.review_dir / "unresolved.csv"

    @property
    def overrides_path(self) -> Path:
        """Hand-written ID corrections. Append-only, authoritative, never overwritten."""
        return PROJECT_ROOT / "overrides" / "film_id_overrides.csv"

    def ensure_dirs(self) -> None:
        for path in (
            self.cache_dir,
            self.external_dir,
            self.processed_dir,
            self.review_dir,
            self.users_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        export_dir=_env_path("LBREC_EXPORT_DIR", "Data"),
        artifacts_dir=_env_path("LBREC_ARTIFACTS_DIR", "artifacts"),
        tmdb_token=os.getenv("LBREC_TMDB_TOKEN") or None,
        tmdb_api_key=os.getenv("LBREC_TMDB_API_KEY") or None,
        tmdb_rps=float(os.getenv("LBREC_TMDB_RPS", "20")),
        default_user=os.getenv("LBREC_USER") or DEFAULT_USER,
    )
