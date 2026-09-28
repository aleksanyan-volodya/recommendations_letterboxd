"""Fetch all of TMDb concurrently, resumably, into a sharded parquet store.

The synchronous client is fine for a few thousand titles and hopeless for a
million: it is latency-bound, not rate-limited, so it idles at ~6 req/s while
the API happily serves 80. Measured on this catalogue:

===============  ==========  ====================================
concurrency      req/s       1.18M films
===============  ==========  ====================================
1 (sync client)  ~6          ~55 hours
8                25.7        12.8 hours
**16**           **84.7**    **3.9 hours**
32               69.4        4.7 hours
===============  ==========  ====================================

Sixteen is the sweet spot; beyond it throughput falls again. No 429s were seen
at any level, but backoff is implemented anyway because a four-hour run will
meet conditions a five-minute probe does not.

Design notes
------------
**Resumable.** Work is written in shards as it completes, and a restart skips
ids already stored. An interrupted run costs only the shard in flight, which
matters when the job is longer than an evening.

**Popularity-ordered, not popularity-filtered.** Shards are fetched most-popular
first so the store is useful after an hour rather than after four. Every film is
eventually fetched -- the ordering decides *when*, never *whether*. Filtering
here would make obscure films unrecommendable by construction, which is the bias
this project exists to remove.

**Raw JSON is not kept.** At 9 KB/film the raw responses are ~11 GB; the
normalised rows are a fraction of that. Re-fetching costs four hours, so the
trade is worth it -- but it does mean a schema change means a re-run, which is
why the appended fields are decided once, up front.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

from lbrec.enrich import MOVIE_APPEND, normalize_movie
from lbrec.tmdb import API_ROOT, Credential, CredentialKind

#: Measured optimum. See the table above.
DEFAULT_CONCURRENCY = 16

#: Films per parquet shard. Small enough that an interrupted run loses little,
#: large enough that the store does not become thousands of tiny files.
SHARD_SIZE = 20_000

#: Statuses that mean "this film is gone", as opposed to "try again".
TERMINAL_STATUSES = frozenset({401, 404})

EXPORT_URL = "https://files.tmdb.org/p/exports/movie_ids_{stamp}.json.gz"


@dataclass
class FetchStats:
    ok: int = 0
    missing: int = 0
    failed: int = 0
    retries: int = 0
    failures: list[int] = field(default_factory=list)

    def merge(self, other: FetchStats) -> None:
        self.ok += other.ok
        self.missing += other.missing
        self.failed += other.failed
        self.retries += other.retries
        self.failures.extend(other.failures)


# --------------------------------------------------------------------------
# the id export
# --------------------------------------------------------------------------
def read_id_export(path: Path, *, include_video: bool = False) -> pd.DataFrame:
    """Read TMDb's daily id dump.

    Carries only ``id``, ``original_title``, ``popularity``, ``adult`` and
    ``video`` -- enough to order and scope the work, not enough to model on.

    ``video=true`` marks concert recordings and direct-to-video items. They are
    excluded by default: the first real recommendation run surfaced BTS concert
    films, and this flag is what distinguishes "not a film" from "obscure film".
    The default export already excludes adult titles.
    """
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("video") and not include_video:
                continue
            records.append(
                {
                    "tmdb_id": int(record["id"]),
                    "original_title": record.get("original_title") or "",
                    "export_popularity": float(record.get("popularity") or 0.0),
                }
            )
    frame = pd.DataFrame.from_records(records)
    # Most popular first: the store becomes useful long before the run ends.
    return frame.sort_values("export_popularity", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def shard_path(store: Path, index: int) -> Path:
    return store / f"films_{index:05d}.parquet"


def stored_ids(store: Path) -> set[int]:
    """Every tmdb_id already written, so a restart does no duplicate work."""
    if not store.exists():
        return set()
    known: set[int] = set()
    for path in sorted(store.glob("films_*.parquet")):
        try:
            known.update(pd.read_parquet(path, columns=["tmdb_id"])["tmdb_id"].astype(int))
        except (OSError, ValueError):
            # A shard truncated by an interrupted write is simply refetched.
            path.unlink(missing_ok=True)
    return known


def pending_ids(catalogue_ids: Iterable[int], store: Path) -> list[int]:
    known = stored_ids(store)
    return [int(i) for i in catalogue_ids if int(i) not in known]


def _chunks(items: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------
def _auth(credential: Credential) -> tuple[dict, dict]:
    headers = {"accept": "application/json"}
    params: dict = {"append_to_response": MOVIE_APPEND, "language": "en-US"}
    if credential.kind is CredentialKind.V4_TOKEN:
        headers["Authorization"] = f"Bearer {credential.value}"
    else:
        params["api_key"] = credential.value
    return headers, params


async def _fetch_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    tmdb_id: int,
    params: dict,
    stats: FetchStats,
    rows: list[dict],
    *,
    max_attempts: int,
) -> None:
    async with semaphore:
        for attempt in range(max_attempts):
            try:
                response = await client.get(f"/movie/{tmdb_id}", params=params)
            except httpx.HTTPError:
                stats.retries += 1
                await asyncio.sleep(2**attempt + random.random())
                continue

            if response.status_code == 200:
                rows.append(normalize_movie(response.json()))
                stats.ok += 1
                return
            if response.status_code in TERMINAL_STATUSES:
                stats.missing += 1
                return
            if response.status_code == 429:
                delay = float(response.headers.get("Retry-After", 2**attempt))
                stats.retries += 1
                await asyncio.sleep(min(delay, 30.0))
                continue
            stats.retries += 1
            await asyncio.sleep(2**attempt + random.random())

        stats.failed += 1
        stats.failures.append(tmdb_id)


async def fetch_shard(
    tmdb_ids: list[int],
    credential: Credential,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int = 4,
) -> tuple[pd.DataFrame, FetchStats]:
    """Fetch one shard's worth of films concurrently."""
    headers, params = _auth(credential)
    semaphore = asyncio.Semaphore(concurrency)
    stats = FetchStats()
    rows: list[dict] = []

    async with httpx.AsyncClient(
        base_url=API_ROOT,
        headers=headers,
        timeout=30.0,
        limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency),
    ) as client:
        await asyncio.gather(
            *(
                _fetch_one(
                    client,
                    semaphore,
                    tmdb_id,
                    params,
                    stats,
                    rows,
                    max_attempts=max_attempts,
                )
                for tmdb_id in tmdb_ids
            )
        )

    return pd.DataFrame.from_records(rows), stats


def write_shard(frame: pd.DataFrame, store: Path, index: int) -> Path:
    """Write a shard atomically, so an interrupted run leaves no partial file."""
    store.mkdir(parents=True, exist_ok=True)
    target = shard_path(store, index)
    temporary = target.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(target)
    return target


def next_shard_index(store: Path) -> int:
    existing = sorted(store.glob("films_*.parquet")) if store.exists() else []
    return len(existing)


def load_store(store: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read every shard back as one frame."""
    shards = sorted(store.glob("films_*.parquet")) if store.exists() else []
    if not shards:
        return pd.DataFrame()
    frames = [pd.read_parquet(path, columns=columns) for path in shards]
    return pd.concat(frames, ignore_index=True).drop_duplicates("tmdb_id", ignore_index=True)
