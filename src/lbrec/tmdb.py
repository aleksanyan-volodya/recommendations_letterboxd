"""A cached, rate-limited TMDb client.

Every response is written to disk keyed by the request, so re-running a
resolution pass costs nothing and the pipeline stays reproducible without
hammering the API. The cache key deliberately excludes the credential, so the
cache survives rotating keys and never stores one.

TMDb has two credential formats and they authenticate differently:

- **v3 API key** -- 32 hex characters, sent as an ``api_key`` query parameter.
- **v4 read access token** -- a JWT, sent as an ``Authorization: Bearer`` header.

Pasting one where the other is expected returns a confusing 401, so the format
is detected from the value rather than trusted from which variable it was put in.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

from lbrec.config import Settings

API_ROOT = "https://api.themoviedb.org/3"

#: Statuses worth caching. A 404 is a stable answer; 429 and 5xx are not.
CACHEABLE_STATUSES = frozenset({200, 404})

_V3_KEY = re.compile(r"[0-9a-f]{32}")


class CredentialKind(StrEnum):
    V3_KEY = "v3_key"
    V4_TOKEN = "v4_token"


class TmdbError(RuntimeError):
    """Raised when TMDb cannot be reached or rejects the credential."""


@dataclass(frozen=True)
class Credential:
    kind: CredentialKind
    value: str

    def describe(self) -> str:
        """Safe for logs: never includes the secret."""
        return f"{self.kind.value} (…{self.value[-4:]})"


def detect_credential(settings: Settings) -> Credential:
    """Work out which credential we were given, whichever variable it landed in.

    A 32-hex-character value is unambiguously a v3 key; anything else is treated
    as a v4 bearer token.
    """
    raw = settings.tmdb_token or settings.tmdb_api_key
    if not raw:
        raise TmdbError(
            "No TMDb credential found. Copy .env.example to .env and set "
            "LBREC_TMDB_TOKEN (get one at https://www.themoviedb.org/settings/api)."
        )
    value = raw.strip().strip('"').strip("'")
    kind = CredentialKind.V3_KEY if _V3_KEY.fullmatch(value) else CredentialKind.V4_TOKEN
    return Credential(kind=kind, value=value)


class RateLimiter:
    """Minimum-interval limiter. TMDb tolerates far more, but there is no hurry."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps if rps > 0 else 0.0
        self._next = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self._interval


class TmdbClient:
    """Minimal TMDb client: disk cache, rate limit, retry on 429 and 5xx."""

    def __init__(
        self,
        settings: Settings,
        *,
        credential: Credential | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        self.credential = credential or detect_credential(settings)
        self.cache_dir = settings.cache_dir / "tmdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(settings.tmdb_rps)
        self._max_retries = max_retries

        headers = {"accept": "application/json"}
        if self.credential.kind is CredentialKind.V4_TOKEN:
            headers["Authorization"] = f"Bearer {self.credential.value}"
        self._client = httpx.Client(
            base_url=API_ROOT, headers=headers, timeout=20.0, transport=transport
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TmdbClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- caching -----------------------------------------------------------
    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        # The credential is never part of the key: the cache must survive key
        # rotation and must not encode a secret.
        canonical = json.dumps(
            {"path": path, "params": {k: v for k, v in sorted(params.items())}},
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    # -- requests ----------------------------------------------------------
    def get(self, path: str, **params: Any) -> dict[str, Any] | None:
        """GET a TMDb endpoint, using the disk cache. ``None`` means 404."""
        params = {k: v for k, v in params.items() if v is not None}
        cache_path = self._cache_path(path, params)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["body"] if cached["status"] == 200 else None

        request_params = dict(params)
        if self.credential.kind is CredentialKind.V3_KEY:
            request_params["api_key"] = self.credential.value

        response = self._request_with_retries(path, request_params)

        if response.status_code == 401:
            raise TmdbError(
                f"TMDb rejected the credential ({self.credential.describe()}). "
                "A v3 key is 32 hex characters; a v4 token is a long JWT."
            )
        if response.status_code not in CACHEABLE_STATUSES:
            raise TmdbError(f"TMDb {path} returned {response.status_code}: {response.text[:200]}")

        body = response.json() if response.status_code == 200 else None
        cache_path.write_text(
            json.dumps({"status": response.status_code, "body": body}, ensure_ascii=False),
            encoding="utf-8",
        )
        return body

    def _request_with_retries(self, path: str, params: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            self._limiter.wait()
            try:
                response = self._client.get(path, params=params)
            except httpx.HTTPError as exc:  # transport failure: retry
                last_error = exc
                time.sleep(2**attempt)
                continue

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                time.sleep(min(retry_after, 30.0))
                continue
            if response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            return response

        raise TmdbError(f"TMDb {path} failed after {self._max_retries} attempts: {last_error}")

    # -- endpoints ---------------------------------------------------------
    def search_movie(
        self, query: str, *, primary_release_year: int | None = None, page: int = 1
    ) -> list[dict[str, Any]]:
        body = self.get(
            "/search/movie",
            query=query,
            primary_release_year=primary_release_year,
            page=page,
            include_adult="true",
            language="en-US",
        )
        return list(body.get("results", [])) if body else []

    def search_tv(
        self, query: str, *, first_air_date_year: int | None = None, page: int = 1
    ) -> list[dict[str, Any]]:
        """Search the TV namespace.

        Letterboxd lists some miniseries and specials, which ``/search/movie``
        structurally cannot find. Results use ``name``/``first_air_date`` rather
        than ``title``/``release_date``.
        """
        body = self.get(
            "/search/tv",
            query=query,
            first_air_date_year=first_air_date_year,
            page=page,
            include_adult="true",
            language="en-US",
        )
        return list(body.get("results", [])) if body else []

    def movie(self, tmdb_id: int, *, append: str = "external_ids") -> dict[str, Any] | None:
        return self.get(f"/movie/{tmdb_id}", append_to_response=append, language="en-US")
