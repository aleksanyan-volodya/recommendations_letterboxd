"""Tests for the TMDb client. No network: every request goes through a mock transport."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from lbrec.config import Settings
from lbrec.tmdb import CredentialKind, TmdbClient, TmdbError, detect_credential

V3_KEY = "a" * 32
V4_TOKEN = "eyJhbGciOiJIUzI1NiJ9.payload.signature"


def make_settings(tmp_path: Path, *, token: str | None = V3_KEY) -> Settings:
    return Settings(
        export_dir=tmp_path / "Data",
        artifacts_dir=tmp_path / "artifacts",
        tmdb_token=token,
        tmdb_api_key=None,
        tmdb_rps=10_000.0,  # keep the limiter out of the way
    )


class RecordingTransport(httpx.MockTransport):
    """Mock transport that remembers every request it served."""

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(wrapped)


def test_detect_credential_reads_the_format_not_the_variable(tmp_path: Path):
    """A v3 key pasted into the token variable must still be recognised as v3."""
    assert detect_credential(make_settings(tmp_path, token=V3_KEY)).kind is CredentialKind.V3_KEY
    assert (
        detect_credential(make_settings(tmp_path, token=V4_TOKEN)).kind is CredentialKind.V4_TOKEN
    )


def test_detect_credential_strips_quotes_and_whitespace(tmp_path: Path):
    credential = detect_credential(make_settings(tmp_path, token=f'  "{V3_KEY}" '))
    assert credential.value == V3_KEY


def test_detect_credential_requires_one(tmp_path: Path):
    with pytest.raises(TmdbError, match="No TMDb credential"):
        detect_credential(make_settings(tmp_path, token=None))


def test_describe_never_leaks_the_secret(tmp_path: Path):
    described = detect_credential(make_settings(tmp_path)).describe()
    assert V3_KEY not in described


def test_v3_key_goes_in_the_query_v4_token_in_the_header(tmp_path: Path):
    # Separate cache dirs: a shared one would serve the second call from disk
    # and record no request at all.
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"id": 550}))
    with TmdbClient(make_settings(tmp_path / "v3", token=V3_KEY), transport=transport) as client:
        client.get("/movie/550")
    assert "api_key" in transport.requests[0].url.params
    assert "Authorization" not in transport.requests[0].headers

    transport = RecordingTransport(lambda r: httpx.Response(200, json={"id": 550}))
    with TmdbClient(make_settings(tmp_path / "v4", token=V4_TOKEN), transport=transport) as client:
        client.get("/movie/550")
    assert "api_key" not in transport.requests[0].url.params
    assert transport.requests[0].headers["Authorization"] == f"Bearer {V4_TOKEN}"


def test_responses_are_cached_on_disk(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"id": 550}))
    settings = make_settings(tmp_path)
    with TmdbClient(settings, transport=transport) as client:
        assert client.get("/movie/550") == {"id": 550}
        assert client.get("/movie/550") == {"id": 550}
    assert len(transport.requests) == 1

    # A fresh client reuses the same on-disk cache.
    transport2 = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(settings, transport=transport2) as client:
        assert client.get("/movie/550") == {"id": 550}
    assert transport2.requests == []


def test_cache_key_ignores_the_credential(tmp_path: Path):
    """Rotating the key must not invalidate the cache, and the key must not be in it."""
    settings = make_settings(tmp_path, token=V3_KEY)
    transport = RecordingTransport(lambda r: httpx.Response(200, json={"id": 550}))
    with TmdbClient(settings, transport=transport) as client:
        client.get("/movie/550")

    other = make_settings(tmp_path, token="b" * 32)
    transport2 = RecordingTransport(lambda r: httpx.Response(500))
    with TmdbClient(other, transport=transport2) as client:
        assert client.get("/movie/550") == {"id": 550}
    assert transport2.requests == []

    cached = next((settings.cache_dir / "tmdb").glob("*.json")).read_text()
    assert V3_KEY not in cached


def test_404_is_cached_as_a_stable_none(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(404, json={"status_code": 34}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        assert client.get("/movie/1") is None
        assert client.get("/movie/1") is None
    assert len(transport.requests) == 1


def test_401_raises_with_a_useful_message(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(401, json={"status_code": 7}))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        with pytest.raises(TmdbError, match="v3 key is 32 hex characters"):
            client.get("/movie/550")


def test_429_is_retried_and_not_cached(tmp_path: Path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"id": 550}),
    ]
    transport = RecordingTransport(lambda r: responses.pop(0))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        assert client.get("/movie/550") == {"id": 550}
    assert len(transport.requests) == 2


def test_persistent_5xx_raises_rather_than_caching_a_bad_answer(tmp_path: Path):
    transport = RecordingTransport(lambda r: httpx.Response(503))
    settings = make_settings(tmp_path)
    with TmdbClient(settings, transport=transport, max_retries=2) as client:
        with pytest.raises(TmdbError):
            client.get("/movie/550")
    assert list((settings.cache_dir / "tmdb").glob("*.json")) == []


def test_search_movie_returns_results_list(tmp_path: Path):
    payload = {"results": [{"id": 389, "title": "12 Angry Men"}]}
    transport = RecordingTransport(lambda r: httpx.Response(200, json=payload))
    with TmdbClient(make_settings(tmp_path), transport=transport) as client:
        results = client.search_movie("12 Angry Men", primary_release_year=1957)
    assert results[0]["id"] == 389
    assert transport.requests[0].url.params["primary_release_year"] == "1957"
