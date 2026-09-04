# recommendations-letterboxd

A personal film recommender trained on a Letterboxd export, built to surface
niche and underrated films rather than just popular and well-rated ones.

## Layout

```
Data/                    raw Letterboxd export, immutable input
artifacts/               everything derived; reproducible, gitignored
  cache/                 raw HTTP responses, keyed by request
  external/              third-party bulk datasets (MovieLens, IMDb)
  processed/             tables the pipeline produces
  review/                CSVs a human is expected to open and edit
overrides/               hand-written corrections; authoritative, never overwritten
src/lbrec/               the package
tests/
```

The export directory is a parameter

## Setup

```bash
uv sync
cp .env.example .env      # then fill in the TMDb token
```

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Ingest the export | `uv run lbrec ingest` | `films_local`, `interactions`, `film_status` |
| Resolve TMDb IDs | `uv run lbrec resolve` | `film_map`, `review/unresolved.csv` |
| Fold in manual fixes | `uv run lbrec resolve-apply` | appends to `overrides/film_id_overrides.csv` |

### ID resolution

The export carries a boxd.it slug, a title and a year with no external ID. Every
dataset the project joins against keys on TMDb or IMDb IDs, so this mapping is
the join key everything else rests on.

Matching is deliberately conservative: a wrong ID is worse than a missing
one. A missing film is visibly absent; a wrong one attaches another
film. Confidence tiers:

| tier | rule | auto-used? |
| --- | --- | --- |
| `exact` | a title form matches verbatim, year within 1 | yes |
| `high` | similarity >= 90, year within 1 | yes |
| `medium` | similarity >= 85, year within 2 | Needs review |
| `unresolved` | nothing met the bar | no |
| `override` | a human decided | yes |

Guards that earned their place on real data:

- **Article variants.** "The Fisher King" vs TMDb's "Fisher King" scores 84.6,
  under threshold. Leading and trailing articles are stripped as extra
  comparison forms, which can only raise a score, never lower one.
- **Sequel markers.** "Drunken Master II" scores 97.1 against "Drunken Master
  III". A fuzzy similarity barely notices a roman numeral. A differing trailing
  number rejects the candidate outright.
- **Literal beats variant.** Article stripping makes "The Stalker" tie with
  "Stalker"; a verbatim title match outranks a variant one.
- **Medium is not auto-used.** In the reference run, one medium match was wrong
  ("Dear Diary" 1993 matched a 1995 film rather than Moretti's *Caro diario*).

Responses are cached under `artifacts/cache/tmdb/`, keyed by request with the
credential excluded, so re-running costs nothing and rotating the key does not
invalidate the cache.

TMDb accepts two credential formats. 32-hex-character v3 key (query
parameter) and a v4 JWT (bearer header). The client detects the format from the value
rather than from which variable it was placed in.

### Television

Letterboxd lists some miniseries and specials, which TMDb keeps in a separate
namespace that `/search/movie` structurally cannot return. The film namespace is
searched first and TV only when it yields no exact match, so a film is never
displaced by a like-named series. Matches carry a `media_type` of `movie` or
`tv`.

**TV is taste signal only.** It informs the user model but never enters the
candidate catalog or evaluation, which keeps MovieLens and the IMDb film dumps
cleanly applicable.

### Manual corrections

Anything the rules will not accept goes to `artifacts/review/unresolved.csv`
with TMDb's top suggestions for context. Fill in `tmdb_id` (and `media_type` if
it is a series), or write a `note` alone to record a deliberate non-match, then
run `lbrec resolve-apply`.

That appends to `overrides/film_id_overrides.csv`, which is committed,
authoritative and never rewritten. Re-running resolution honours overrides
without touching the API, so hand-checked work is never lost.

## What the export actually contains

Numbers below are from the reference export (873 watched films).

| kind | rows | reliability |
| --- | --- | --- |
| `watched` | 873 | high |
| `watchlist` | 865 | high, but availability-biased |
| `rating` | 788 | the primary label; full 1-5 scale, mean 3.20, sd 0.911 |
| `like` | 102 | binary, sits on top of a rating |
| `diary` | 84 | **low** -- optional enrichment, see below |
| `list_entry` | 38 | low, idiosyncratic |

## Traps in the export format

Four things that silently corrupt naive use of this data. All are handled in
`src/lbrec/letterboxd.py` and pinned by tests.

1. **`diary.csv` is keyed by diary-entry URI, not film URI.** Its 6-char slugs
   share no values with `watched.csv`. Diary rows join on `(title, year)`.
2. **List files are three blocks in one CSV** -- magic line, list metadata,
   blank line, then the films -- with CRLF endings.
3. **Every `Date` column is a logging date, not a viewing date.** 42% of the
   reference export's watches are stamped in the two days after the account was
   created, because they were backfilled. The only true chronology is
   `diary.Watched Date`, which most users do not keep. *Do not build temporal
   train/test splits on these dates* -- they measure logging order, not taste
   over time.
4. **`watched.csv` is not a complete record of what was seen.** When Letterboxd
   deletes or merges a film entry, the watch record vanishes from `watched.csv`
   while the diary entry survives under `deleted/` and the watchlist entry stays
   put. So `seen` is the union of watched, rated and diarised, and
   `watchlist_pending` is the watchlist with those removed. Anything seen is
   excluded from both recommendation and evaluation.

`deleted/` and `orphaned/` entries are real activity, so they are ingested and
flagged with `deleted=True` rather than dropped.

## Diary is optional

Diary coverage is sparse here (84 rows against 788 ratings, and only really
active since mid-2026) and absent entirely for most Letterboxd users, who mark
films watched and rate them without keeping a diary. It is ingested because
`rewatch` and real watch dates are free when present, but **nothing downstream
may require it**. `coverage_report` prints per-kind coverage on every ingest so
this stays visible when choosing features.

## Outputs

- `films_local.parquet` -- one row per distinct film in the export.
- `interactions.parquet` -- long format, one row per (film, interaction). Every
  signal lands here with the same schema, so downstream code has a single door.
  Ingesting a signal is not weighting it; that is a Phase 1 decision made
  against the coverage report.
- `film_status.parquet` -- one authoritative row per film: `seen`, `rating`,
  `liked`, `on_watchlist`, `watchlist_pending`. This is what modelling consumes.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
