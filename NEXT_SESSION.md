# Handoff prompt — Letterboxd recommender, phase 2

Paste this into a new session started in this repo.

---

## The goal

Build a recommender that any person can use by exporting their own Letterboxd
data and handing it over. Not a model of my taste — a model of taste, which my
ratings are one input to. It must work identically for me, my sister, or a
stranger in Tashkent. I will use it locally with friends (no redistribution, no
commercial use), but **nothing may be hardcoded to my personal data**.

### The non-negotiable constraint

Niche and underrated films must be surfaced **fairly relative to quality, not
relative to obscurity**. Handling this properly matters more to me than raw
accuracy on popular titles. It is normal to discover a film that someone would not like, it's a part of the discovery process. Equally: I want the best result, not the quick one —
a "fair" model that ranks badly is not a solution.

### What changes in this phase

1. **Recommend only films that exist on Letterboxd**, not from the full 1.15M
   TMDb catalogue. The candidate universe shrinks to what Letterboxd knows.
2. **Augment with outside data freely as you want.** Other databases for synopses, metadata,
   tags, anything — augmentation is encouraged, it just must not change the
   candidate universe.
3. **Go multimodal and multi-signal**, specifically:
   - film synopses / abstracts (text embeddings, not just TF-IDF)
   - a **small set of high-signal raters** — people who have seen very many
     films, or who are widely followed — treated differently from the average
     rater
   - **user archetypes**, in the spirit of the Netflix Prize: latent "typical
     viewers" that a new person is expressed as a mixture of
   - **tags** — both existing tag data and tags inferred for films that have none
   - **language** and other categorical facts as real signal, not filler

## Where the project is now

196 tests pass, ruff clean, working tree clean at `6a54b25`.

`TODO.md` (gitignored) is the living record — **read §0 first**, it has the
current numbers and the two protocol mistakes that cost redesigns. Treat the
backlog sections as a menu, not a plan.

### Architecture (settled, don't relitigate)

`src/lbrec/generalise.py` is the live architecture:

```python
fit_global(ratings_of_many_users)   # once, offline, shared by everyone
score_user(given_items, given_ratings, target_items)   # history as input
rank_user(...)                      # optional: ranking ≠ prediction
```

Nothing is refitted per person. A new user's ratings are an *argument*, not a
training set. Models: `UserMean`, `ItemMean`, `BiasModel`, `BiasedMF`,
`ContentTower`, `StratifiedRanker`.

The older per-user architecture (`models.py`, `evaluate.py`) is kept only as a
reference point. Don't build on it.

### Pipeline

`ingest → resolve → enrich → movielens → catalogue → generalise / cold-items`,
all via the `lbrec` CLI. Per-user tables live under `artifacts/users/<user>/`,
shared film facts under `artifacts/processed/`.

## Data actually on disk — verified, not remembered

| source | what it is | size |
|---|---|---|
| `artifacts/external/letterboxd-samlearner.zip` | `ratings_export.csv`: **11,078,167 ratings, 7,477 users, 286,072 films**, 1–10 scale. Plus `movie_data.csv` (272,878 films) and `users_export.csv` | 197 MB |
| `artifacts/external/movielens/` | ml-32m: 32M ratings, 200,948 users, 87,585 films. **`tags.parquet`: 2,000,072 tags, 51,323 films, 140,981 distinct tags** | |
| `artifacts/external/tmdb_films/` | bulk TMDb store, 1,174,109 films, full metadata (keywords, cast, crew) | |
| `artifacts/processed/catalogue.parquet` | 1,151,334 recommendable films | |
| `artifacts/users/me/` | my export: 1,741 films, 790 rated | |

The MovieLens **tag** data has never been used. Neither had the Letterboxd
ratings until the last hour of the previous session — I had been reading only the
metadata out of that archive.

## Findings that must not be re-derived

Each of these cost real measurement. Do not re-litigate without new evidence.

1. **Almost all predictive power is in bias terms.** `μ + b_i + b_u` reaches
   0.873 RMSE on held-out users; 64 latent factors add 0.003, and at 10 given
   ratings `biased_mf` is *worse* than plain `bias`. **This is a warning for the
   "user archetypes" idea** — latent structure has so far bought nothing here.
   It needs to earn its place against that baseline, not be assumed.
2. **Ensembling the collaborative family is dead.** Per-user error correlations
   0.98–0.995, oracle headroom 0.93%. Only a model using *different information*
   broke that: the content tower correlates 0.22 with the factorisation.
3. **Popularity bias is in what is *knowable*, not only in ratings.** Metadata
   richness is monotone in popularity: keyword coverage 18% at zero votes → 100%
   at 10k+; directors 81% → 100%. A keyword-heavy model is secretly a model of
   famous films. **This is the main risk for the multimodal plan** — synopsis
   embeddings and inferred tags are only fair if coverage in the tail is checked
   first.
4. **`b_i` is not comparable across crowd-rated and crowd-less films.** Measured
   for one, guessed or absent for the other. Discounting it fails (fairness only
   arrives at weight exactly 0, which destroys ranking). `StratifiedRanker`
   fixes it by ranking each group separately and interleaving proportionally:
   fair at every history size (0.93–0.98 of deserved tail share) *and* the best
   ranker we have. Spearman went **up**, because the guessed `b_i` was also wrong.
5. **My own library is not obscure.** 749 of 786 rated films sit in the top band
   of the real catalogue. Any claim about my taste being niche is false.
6. **`TextBlock` once dropped its vectoriser settings** because sklearn's
   `get_params()` only reports explicitly named `__init__` args, and
   `ColumnTransformer` clones transformers before fitting. Every content number
   was wrong for weeks. Lesson: never take `**kwargs` in an sklearn-compatible
   estimator.

## Two protocol rules, learned the hard way

- **Never score crowd-less films in isolation.** With `b_i` absent for every
  candidate, a missing-bias strategy has nothing to differ about and distinct
  models report identical numbers. Popularity bias is about what a film *loses
  to*, so candidates must mix known and unknown. Pinned by
  `test_missing_bias_strategies_cannot_differ_on_cold_films_alone`.
- **Ranking and prediction are different jobs.** Any damped or interleaved score
  is no longer a rating estimate. RMSE comes from `score_user`, ranking metrics
  from `rank_user`.

The objective is already encoded in `generalise.frontier()`: **constraint** —
crowd-less films reach a user's top 10 at ≥90% of the rate their own ratings
deserve; **objective** — subject to that, maximise Spearman. Overshooting is
reported but never rewarded. Use it; don't eyeball tables.

## The open problem this phase inherits

The stratified ranker's fairness quota is the crowd-less share *of the candidate
set*. Over the full 1.15M catalogue that share is ~93%, so a proportional top 10
would be ~9 films nobody has ever rated — which is promoting obscurity, the
opposite of the goal.

The Letterboxd ratings solve this by defining a pool with a verified audience:

| threshold | pool | MovieLens-cold | median TMDb votes |
|---|---:|---:|---:|
| LB ≥1 | 256,108 | 70.1% | 5 |
| **LB ≥5** | **98,647** | **39.7%** | **20** |
| LB ≥10 | 63,897 | 25.6% | 38 |

`LB ≥5` is 39.7% crowd-less — inside the 25–47% band the ranker was validated
across, so the fairness guarantee transfers. Half that pool sits at ≤19 TMDb
votes and 2,595 films have zero, so it is genuinely tail-weighted rather than a
popularity filter in disguise.

**Known tension with the new goal, needs a decision:** the dump is a ~Dec 2020
scrape. It effectively ends in 2021 (2022: 1,587 films; 2023: 172; 2024: 64).
Today's Letterboxd has far more than 286k films. So "only films on Letterboxd"
currently means "films on Letterboxd as of 2020". Letterboxd has no public API
and we are not scraping it. Either accept the staleness, or find another way to
establish what is on Letterboxd now — decide this explicitly and early, because
it defines the candidate universe.

## What we do not have

Be honest about this rather than inventing it:

- **No critics data.** No Sight & Sound, TSPDT, Criterion, Metacritic or
  RT corpus is on disk. The nearest available proxy for "a small group of people
  who have seen a great many films" is the Letterboxd dump's heaviest raters —
  7,477 users averaging ~1,482 ratings each, already a cinephile-skewed
  population. If real critic lists are wanted, they must be sourced first, and
  licence-checked.
- **No live Letterboxd access**, as above.
- **No poster/image or audio data**, so "multimodal" currently means text +
  structured metadata + behaviour. Say so plainly rather than overclaiming.

## How I want to work

- Step by step, no skips, no shortcuts. Clean project. I read the code.
- Tests alongside the code — they have caught real bugs here repeatedly, and a
  test that encodes *why* a thing is done has paid for itself more than once.
- **Give me the command for any long job** (anything over a couple of minutes); I
  run it in my own terminal and tell you when the CSV is written. Do not launch
  it yourself.
- Report results honestly, including when something got worse or when an earlier
  claim of yours turns out wrong. That has happened several times here and it was
  always more useful than a clean story.
- Measure before believing. Several confident predictions in this project were
  wrong — including that predicting `b_i` from content would be the *fair*
  choice, when it turned out to be the most tail-suppressing option of all.

## Suggested order of work (argue with it if you disagree)

1. **Fix the candidate universe.** Decide the Letterboxd-only question above,
   then build the pool as a first-class artifact with a `lbrec` command. Verify
   the crowd-less share lands in the validated 25–47% band.
2. **Use the Letterboxd ratings as a second crowd signal.** 256k films get a
   real `b_i` instead of 84k, which shrinks the guessing problem directly. Then
   re-run `cold-items` and see whether `StratifiedRanker` still wins.
3. **Run the generalisation protocol on Letterboxd users instead of MovieLens
   users.** Nothing has ever checked whether these results hold across
   populations, and this is a cinephile population — closer to the real users.
   7,477 users is few, so watch for it.
4. **Then multimodal**, hardest-first on coverage grounds: synopsis embeddings,
   then tags (real where available, inferred where not), then language and
   categorical signal. Check tail coverage of every new feature *before*
   measuring its accuracy — finding 3 above is why.
5. **Archetypes last**, and only with finding 1 in hand as the bar to beat.

Start by reading `TODO.md` §0 and confirming the numbers above still match the
artifacts on disk. If they don't, say so before building anything.
