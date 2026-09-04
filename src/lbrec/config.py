"""Project paths and settings.

Everything derived lives under ``artifacts/`` and is reproducible from the raw
Letterboxd export plus the external sources. The raw export directory is a
parameter rather than a constant, because the eventual web app has to ingest
other people's exports the same way it ingests ours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one run."""

    export_dir: Path
    artifacts_dir: Path
    tmdb_token: str | None
    tmdb_api_key: str | None
    tmdb_rps: float

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
    def films_local_path(self) -> Path:
        return self.processed_dir / "films_local.parquet"

    @property
    def interactions_path(self) -> Path:
        return self.processed_dir / "interactions.parquet"

    @property
    def film_status_path(self) -> Path:
        """One row per film: seen, rating, liked, watchlist_pending."""
        return self.processed_dir / "film_status.parquet"

    @property
    def film_map_path(self) -> Path:
        return self.processed_dir / "film_map.parquet"

    @property
    def unresolved_path(self) -> Path:
        """Written for the human review pass."""
        return self.review_dir / "unresolved.csv"

    @property
    def overrides_path(self) -> Path:
        """Hand-written ID corrections. Append-only, authoritative, never overwritten."""
        return PROJECT_ROOT / "overrides" / "film_id_overrides.csv"

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.external_dir, self.processed_dir, self.review_dir):
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
    )
