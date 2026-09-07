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
| Fetch TMDb metadata | `uv run lbrec enrich` | `films_tmdb` |
| Prepare MovieLens | `uv run lbrec movielens` | `external/movielens/*.parquet` |
| Report taste diagnostics | `uv run lbrec profile` | printed |
| Cross-validate models | `uv run lbrec evaluate` | printed |
| Interactive dashboard | `uv run lbrec dashboard` | Streamlit app |

### Dashboard

`uv run lbrec dashboard` (equivalently `uv run streamlit run src/lbrec/dashboard/app.py`)
launches a local Streamlit app over whatever artifacts are already on disk:

- **Overview** -- rating distribution, catalogue-vs-rated popularity, rating by
  genre/decade, most-watched directors.
- **Taste profile** -- the `lbrec profile` diagnostics (crowd-score and
  popularity correlations, popularity deciles, tail share, watched-vs-wanted)
  as charts.
- **Model playground** -- cross-validate any combination of models with live
  hyperparameter sliders (SVD components, XGBoost params, k in kNN, hybrid
  inner folds, CV folds/repeats/seed), reproducing what `lbrec evaluate`
  reports but interactively.
- **Explore predictions** -- actual-vs-predicted scatter and largest misses for
  one model from the last playground run, by title.

It reads the parquet artifacts directly and computes nothing the pipeline
doesn't already compute -- run `ingest`/`resolve`/`enrich` (and
`movielens`/`factors` for the collaborative and hybrid models) first.

### Evaluation

`lbrec evaluate` runs k-fold cross-validation over one user's ratings and
reports **every metric per popularity decile as well as overall**. That slice is
the point: a model at RMSE 0.72 overall and 0.95 in the bottom deciles is
failing at exactly what this project exists for, and the headline number hides
it.

Folds are random and seeded. A temporal split is not offered, because the only
dates in a Letterboxd export are logging dates -- see the export traps above --
so splitting on them would measure data-entry order, not taste over time.

Models implement one interface (`fit(features, ratings)` / `predict(features)`),
so a ridge, a factorisation or an LLM reranker are substitutions rather than
rewrites. Every transformer is fitted inside the fold: with a few hundred labels,
a vocabulary or scaler fitted over the whole set leaks and inflates every score.

Two baselines are always included. `global_mean` is the floor. `crowd_score` --
the crowd's average rating, rescaled to this user -- is the bar that matters: a
personal model that cannot beat it has learned nothing personal.

### The two legs, and two traps

`content_ridge` works from TMDb metadata and covers every film.
`collaborative_fold` (`lbrec factors`) holds MovieLens item factors fixed and
solves only for the user's position in that space -- the cold-start-user fold-in,
which is well determined by a few hundred ratings where learning a representation
from them would not be. `hybrid` stacks the two.

Both legs shipped with a bug that produced plausible output rather than an error.
Each now has a regression test:

1. **Item-centring removes the quality signal.** Ratings are centred per item
   before factorisation so popularity does not dominate the components -- but
   centring also removes *how well* a film is rated, which is the strongest
   collaborative signal there is and the one `crowd_score` lives on. The item
   mean must be carried alongside the latent directions. Restoring it moved the
   collaborative leg from 0.848 to 0.745 RMSE on the reference data.
2. **Stacking weights must be fitted out of fold.** Ridge fits its own training
   fold far more closely than the collaborative leg fits its, so weights fitted
   on in-sample predictions hand the ridge everything and the blend silently
   collapses to one model. Weights come from an inner cross-validation.

### Enrichment

One request per title with genres, keywords, credits and external IDs appended,
so the whole content feature set arrives in a single round trip. Two fields
matter out of proportion to the rest:

- **`imdb_id`** -- the join key to the IMDb bulk datasets and MovieLens
  `links.csv`. Without it a title cannot be attached to any external data.
- **`vote_count`** -- the popularity variable. Every debiasing technique in the
  plan is a function of it, so it is a first-class column from the start.

Film and TV payloads name the same ideas differently (`title`/`name`,
`release_date`/`first_air_date`, keywords under `keywords`/`results`, directors
vs `created_by`). Both are normalised onto one schema; `media_type` keeps them
distinguishable.

Note that list columns (`genres`, `keywords`, `cast`, ...) come back from
parquet as numpy arrays rather than Python lists.

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

## MovieLens

`uv run lbrec movielens` converts the downloaded archive to parquet and measures
what it covers. MovieLens substitutes for the user base this project does not
have: 32M ratings from 200,948 users over 87,585 films (to October 2023).

Download `ml-32m.zip` from <https://grouplens.org/datasets/movielens/> into
`artifacts/external/`. Leave it zipped -- the loader reads the archive directly,
so the pipeline stays reproducible from the original download. Note that
`files.grouplens.org` has been serving an expired TLS certificate since 28
August 2026, so a browser will warn before the download starts.

**Licence: research use, free redistribution under the same terms, no commercial
or revenue-bearing use without permission from GroupLens.** A free public site is
within terms; anything revenue-bearing is not.

`links.csv` has a trap: **MovieLens stores IMDb ids as bare integers with the
`tt` prefix and leading zeros stripped**, so `114709` means `tt0114709`. Joining
on the raw column matches nothing, silently. `imdb_tt()` restores them.

### Presence is not signal

The coverage report is deliberately sliced by popularity, because the headline
number hides the thing that decides the architecture. A film can be *in*
MovieLens and still carry no usable collaborative signal: an item with nine
ratings out of 200k users yields a noise vector from any factorisation.

So the report shows both coverage **and** median ratings per film per decile.
Where median support collapses, the collaborative leg cannot serve those films
and content features have to, which is what the hybrid design and its coverage
gate exist for. Run it on your own export to find where that boundary falls.

## Taste profile

`uv run lbrec profile` reports how one user's ratings relate to catalogue
popularity. These are **per-user quantities, not project constants** -- one
person's watchlist is long and full of obscurities, another's is short and
entirely mainstream, and the two need different handling -- so they are computed
from whichever export is loaded rather than assumed.

Three of them decide how modelling should go for that user:

| quantity | what it decides |
| --- | --- |
| `rho(rating, crowd score)` | the real baseline. If a model cannot beat "predict what everyone else thought", it has learned nothing personal. |
| `rho(rating, log vote_count)` | how much of this taste fame already explains. Near zero means a popularity-biased model is not merely suboptimal, it is wrong. |
| tail share | whether the history has a long tail at all. If it all sits in the top deciles there is nothing to surface, and the debiasing work has no purchase. |

The RMSE floor (predicting the user's mean) is also reported, but it is the easy
bar; the crowd baseline is the one that matters.

Deciles are cut against the **catalogue**, not against the user's own films, so
"decile 1" means the same thing for everybody.

`Watched versus wanted` compares the popularity of what a user has rated against
what is still on their watchlist. Watchlists are usually assumed to be
availability-biased toward well-known titles, but that is not universal, and
which way it runs for a given user decides whether their watchlist is safe to
use as a held-out positive set.

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
