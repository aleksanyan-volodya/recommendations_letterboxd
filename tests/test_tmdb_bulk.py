"""Tests for bulk TMDb enrichment.

The properties that matter here are about a four-hour job surviving contact with
reality: it must resume without redoing work, never leave a half-written shard,
and never quietly drop films. No network -- every request goes through a mock
transport.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from lbrec.tmdb import Credential, CredentialKind
from lbrec.tmdb_bulk import (
    fetch_shard,
    load_store,
    next_shard_index,
    pending_ids,
    read_id_export,
    stored_ids,
    write_shard,
)

V3 = Credential(kind=CredentialKind.V3_KEY, value="a" * 32)

EXPORT_ROWS = [
    {"adult": False, "id": 1, "original_title": "Popular", "popularity": 90.0, "video": False},
    {"adult": False, "id": 2, "original_title": "Obscure", "popularity": 0.1, "video": False},
    {"adult": False, "id": 3, "original_title": "Middling", "popularity": 5.0, "video": False},
    {"adult": False, "id": 4, "original_title": "Concert", "popularity": 50.0, "video": True},
]


def payload(tmdb_id: int) -> dict:
    return {
        "id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "release_date": "1999-01-01",
        "vote_count": 100,
        "vote_average": 7.0,
        "genres": [{"id": 18, "name": "Drama"}],
        "keywords": {"keywords": [{"id": 1, "name": "kw"}]},
        "credits": {"cast": [], "crew": []},
        "external_ids": {"imdb_id": f"tt{tmdb_id:07d}"},
    }


@pytest.fixture
def export_file(tmp_path: Path) -> Path:
    path = tmp_path / "ids.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in EXPORT_ROWS:
            handle.write(json.dumps(row) + "\n")
    return path


# --------------------------------------------------------------------------
# the id export
# --------------------------------------------------------------------------
def test_export_excludes_video_entries(export_file: Path):
    """video=true marks concert recordings, which polluted the first real run."""
    films = read_id_export(export_file)
    assert films["tmdb_id"].tolist() == [1, 3, 2]  # popularity-ordered
    assert 4 not in set(films["tmdb_id"])


def test_export_can_keep_video_entries_when_asked(export_file: Path):
    films = read_id_export(export_file, include_video=True)
    assert set(films["tmdb_id"]) == {1, 2, 3, 4}


def test_export_is_ordered_most_popular_first(export_file: Path):
    """An ordering, not a filter: it decides when a film is fetched, not whether."""
    films = read_id_export(export_file)
    assert films["export_popularity"].is_monotonic_decreasing
    assert len(films) == 3  # every non-video film is still present


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------
def run_fetch(ids, handler, **kwargs):
    transport = httpx.MockTransport(handler)

    async def go():
        import lbrec.tmdb_bulk as bulk

        original = httpx.AsyncClient

        def patched(*args, **inner):
            inner["transport"] = transport
            return original(*args, **inner)

        bulk.httpx.AsyncClient = patched
        try:
            return await fetch_shard(ids, V3, **kwargs)
        finally:
            bulk.httpx.AsyncClient = original

    return asyncio.run(go())


def test_fetch_normalises_every_film():
    frame, stats = run_fetch(
        [1, 2, 3], lambda r: httpx.Response(200, json=payload(int(r.url.path.split("/")[-1])))
    )
    assert stats.ok == 3
    assert set(frame["tmdb_id"]) == {1, 2, 3}
    assert frame["imdb_id"].notna().all()


def test_deleted_films_are_counted_not_retried():
    """A 404 is a stable answer; retrying it would waste hours across a million films."""
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404, json={"status_code": 34})

    frame, stats = run_fetch([1, 2], handler)
    assert stats.missing == 2
    assert stats.ok == 0
    assert frame.empty
    assert len(calls) == 2  # one attempt each, no retries


def test_rate_limiting_is_retried_then_succeeds():
    responses = {
        1: [httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json=payload(1))]
    }

    def handler(request):
        tmdb_id = int(request.url.path.split("/")[-1])
        return responses[tmdb_id].pop(0)

    frame, stats = run_fetch([1], handler)
    assert stats.ok == 1
    assert stats.retries == 1


def test_films_that_never_succeed_are_recorded_not_silently_dropped():
    """A four-hour run must say which films it failed on, not just lose them."""
    frame, stats = run_fetch([7], lambda r: httpx.Response(500), max_attempts=2)
    assert stats.failed == 1
    assert stats.failures == [7]
    assert frame.empty


# --------------------------------------------------------------------------
# the resumable store
# --------------------------------------------------------------------------
def test_store_round_trips(tmp_path: Path):
    store = tmp_path / "store"
    write_shard(pd.DataFrame({"tmdb_id": [1, 2], "title": ["a", "b"]}), store, 0)
    write_shard(pd.DataFrame({"tmdb_id": [3], "title": ["c"]}), store, 1)
    assert stored_ids(store) == {1, 2, 3}
    assert len(load_store(store)) == 3


def test_pending_skips_what_is_already_stored(tmp_path: Path):
    """The resumability guarantee: a restart must not refetch."""
    store = tmp_path / "store"
    write_shard(pd.DataFrame({"tmdb_id": [1, 2]}), store, 0)
    assert pending_ids([1, 2, 3, 4], store) == [3, 4]


def test_shard_writes_are_atomic(tmp_path: Path):
    """No .tmp file may survive, or a restart would read a truncated shard."""
    store = tmp_path / "store"
    write_shard(pd.DataFrame({"tmdb_id": [1]}), store, 0)
    assert list(store.glob("*.tmp")) == []
    assert (store / "films_00000.parquet").exists()


def test_a_corrupt_shard_is_discarded_and_refetched(tmp_path: Path):
    """An interrupted write leaves an unreadable file; it must not wedge the run."""
    store = tmp_path / "store"
    write_shard(pd.DataFrame({"tmdb_id": [1, 2]}), store, 0)
    (store / "films_00001.parquet").write_bytes(b"not parquet")

    assert stored_ids(store) == {1, 2}
    assert not (store / "films_00001.parquet").exists()
    assert pending_ids([1, 2, 3], store) == [3]


def test_next_shard_index_continues_after_existing_shards(tmp_path: Path):
    store = tmp_path / "store"
    assert next_shard_index(store) == 0
    write_shard(pd.DataFrame({"tmdb_id": [1]}), store, 0)
    assert next_shard_index(store) == 1


def test_load_store_deduplicates(tmp_path: Path):
    """Overlapping shards can happen after a crash; the store must stay one row per film."""
    store = tmp_path / "store"
    write_shard(pd.DataFrame({"tmdb_id": [1, 2]}), store, 0)
    write_shard(pd.DataFrame({"tmdb_id": [2, 3]}), store, 1)
    assert sorted(load_store(store)["tmdb_id"]) == [1, 2, 3]


def test_empty_store_is_not_an_error(tmp_path: Path):
    assert stored_ids(tmp_path / "nothing") == set()
    assert load_store(tmp_path / "nothing").empty
