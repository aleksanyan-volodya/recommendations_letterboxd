"""Command line entry points.

Each command is one reproducible step of the pipeline. Steps write to
``artifacts/`` and never modify the raw export.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from lbrec.config import get_settings
from lbrec.enrich import enrich_films, popularity_report
from lbrec.evaluate import evaluate, tail_summary
from lbrec.features import build_film_features
from lbrec.letterboxd import coverage_report, film_status, load_export
from lbrec.models import (
    CONTENT_ONLY,
    StackedEnsemble,
    available_models,
    build_model,
    default_models,
)
from lbrec.movielens import build_item_factors, coverage_by_popularity, link_films, load, prepare
from lbrec.profile import build_profile, popularity_gap
from lbrec.resolve import (
    OVERRIDE,
    REVIEW_CONFIDENCES,
    TRUSTED_CONFIDENCES,
    append_overrides,
    build_review_table,
    duplicate_ids,
    load_overrides,
    read_reviewed,
    resolve_films,
)
from lbrec.tmdb import TmdbClient, TmdbError

app = typer.Typer(add_completion=False, help="Letterboxd recommender pipeline.")
console = Console()


@app.callback()
def main() -> None:
    """Letterboxd recommender pipeline.

    Present so Typer keeps subcommand dispatch even while only one command exists
    More pipeline steps land here as they are built.
    """


def _render(df, title: str) -> None:
    table = Table(title=title, title_justify="left", header_style="bold")
    for column in df.columns:
        table.add_column(str(column))
    for row in df.itertuples(index=False):
        table.add_row(*("" if value is None else str(value) for value in row))
    console.print(table)


@app.command()
def ingest(
    user: str | None = typer.Option(None, help="Label for whose export this is (default: 'me')."),
    export_dir: Path | None = typer.Option(
        None, help="Letterboxd export directory. Defaults to the configured export dir."
    ),
    include_deleted: bool = typer.Option(
        True, help="Include deleted/ and orphaned/ entries, flagged instead of dropped."
    ),
) -> None:
    """Parse one Letterboxd export into that user's films, interactions and status tables."""
    settings = get_settings()
    settings.ensure_dirs()
    paths = settings.user(user).ensure()
    source = export_dir or settings.export_dir

    if not source.exists():
        raise typer.BadParameter(f"Export directory not found: {source}")

    export = load_export(source, include_deleted=include_deleted)
    if export.interactions.empty:
        console.print(f"[red]No interactions parsed from {source}.[/red]")
        raise typer.Exit(code=1)

    status = film_status(export.interactions)
    export.films_local.to_parquet(paths.films_local, index=False)
    export.interactions.to_parquet(paths.interactions, index=False)
    status.to_parquet(paths.film_status, index=False)

    _render(coverage_report(export.interactions), "Signal coverage")

    films = export.films_local
    synthetic = int(films["synthetic_key"].sum())
    console.print(
        f"[bold]{len(films)}[/bold] distinct films, "
        f"[bold]{len(export.interactions)}[/bold] interactions "
        f"({synthetic} film(s) without a Letterboxd film URI, keyed by title+year)"
    )
    console.print(
        f"[bold]{int(status['seen'].sum())}[/bold] seen, "
        f"[bold]{int(status['rating'].notna().sum())}[/bold] rated, "
        f"[bold]{int(status['watchlist_pending'].sum())}[/bold] watchlist pending "
        f"({int((status['on_watchlist'] & status['seen']).sum())} watchlist entries dropped as "
        f"already seen)"
    )
    for path in (paths.films_local, paths.interactions, paths.film_status):
        console.print(f"wrote {path.relative_to(settings.artifacts_dir.parent)}")
    console.print(
        "[dim]Note: every Date column is a logging date, not a viewing date. "
        "Do not build temporal splits on them.[/dim]"
    )


@app.command()
def resolve(
    limit: int | None = typer.Option(None, help="Resolve only the first N films (for a dry run)."),
    review: bool = typer.Option(
        True, help="Write unresolved films to artifacts/review/unresolved.csv."
    ),
) -> None:
    """Map every Letterboxd film to a TMDb ID, writing film_map.parquet.

    Responses are cached on disk, so re-running is cheap and only newly seen
    films cost an API call.
    """
    settings = get_settings()
    settings.ensure_dirs()

    users = settings.known_users()
    if not users:
        raise typer.BadParameter("no ingested exports found. Run `lbrec ingest` first.")

    # Resolve the union across everyone: film identity is not per-user, so one
    # pass serves every export and shares the HTTP cache and the overrides.
    films = (
        pd.concat(
            [pd.read_parquet(settings.user(u).films_local) for u in users], ignore_index=True
        )[["film_key", "title", "year"]]
        .drop_duplicates("film_key")
        .sort_values("film_key", ignore_index=True)
    )
    console.print(
        f"resolving {len(films)} distinct films across {len(users)} export(s): {', '.join(users)}"
    )
    if limit is not None:
        films = films.head(limit)

    overrides = load_overrides(settings.overrides_path)
    if overrides:
        console.print(f"loaded [bold]{len(overrides)}[/bold] manual override(s)")

    try:
        with TmdbClient(settings) as client:
            console.print(f"TMDb credential: {client.credential.describe()}")
            with tqdm(total=len(films), desc="resolving", unit="film") as bar:
                film_map = resolve_films(client, films, overrides, progress=bar)
            if review:
                review_table = build_review_table(client, film_map)
                review_table.to_csv(settings.unresolved_path, index=False)
    except TmdbError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    film_map.to_parquet(settings.film_map_path, index=False)

    counts = (
        film_map["confidence"].value_counts().rename_axis("confidence").reset_index(name="films")
    )
    counts["share"] = (counts["films"] / len(film_map)).map("{:.1%}".format)
    _render(counts, "Match confidence")

    trusted = int(film_map["confidence"].isin(TRUSTED_CONFIDENCES).sum())
    # Films already decided by hand are not awaiting anyone, including those
    # recorded as deliberately unmatchable.
    pending = int(
        (film_map["confidence"].isin(REVIEW_CONFIDENCES) & (film_map["source"] != OVERRIDE)).sum()
    )
    decided_absent = len(film_map) - trusted - pending
    console.print(
        f"[bold]{trusted}[/bold] / {len(film_map)} resolved, "
        f"[bold]{pending}[/bold] awaiting a human"
        + (f", {decided_absent} recorded as not on TMDb" if decided_absent else "")
    )
    console.print(f"wrote {settings.film_map_path.relative_to(settings.artifacts_dir.parent)}")

    collisions = duplicate_ids(film_map)
    if not collisions.empty:
        console.print(
            f"[yellow]{collisions['tmdb_id'].nunique()} TMDb id(s) claimed by more than one "
            "film -- these would double-count:[/yellow]"
        )
        _render(collisions[["film_key", "title", "year", "tmdb_id", "confidence"]], "Collisions")

    if review and pending:
        console.print(
            f"wrote {settings.unresolved_path.relative_to(settings.artifacts_dir.parent)} "
            "-- fill in the tmdb_id column, then run `lbrec resolve-apply`"
        )


@app.command()
def enrich() -> None:
    """Fetch TMDb metadata for every resolved title, writing films_tmdb.parquet."""
    settings = get_settings()
    settings.ensure_dirs()

    if not settings.film_map_path.exists():
        raise typer.BadParameter("film_map.parquet not found. Run `lbrec resolve` first.")

    film_map = pd.read_parquet(settings.film_map_path)
    total = len(film_map[film_map["tmdb_id"].notna()][["tmdb_id", "media_type"]].drop_duplicates())

    try:
        with TmdbClient(settings) as client:
            with tqdm(total=total, desc="enriching", unit="title") as bar:
                films = enrich_films(client, film_map, progress=bar)
    except TmdbError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    films.to_parquet(settings.films_tmdb_path, index=False)

    missing = films.attrs.get("missing_tmdb_ids") or []
    with_imdb = int(films["imdb_id"].notna().sum())
    console.print(
        f"[bold]{len(films)}[/bold] titles enriched, "
        f"[bold]{with_imdb}[/bold] with an IMDb id "
        f"({len(films) - with_imdb} without)"
    )
    if missing:
        console.print(f"[yellow]{len(missing)} TMDb id(s) returned 404: {missing[:10]}[/yellow]")

    _render(popularity_report(films), "Catalogue by popularity decile (TMDb vote_count)")
    console.print(f"wrote {settings.films_tmdb_path.relative_to(settings.artifacts_dir.parent)}")


@app.command()
def movielens(
    archive: Path | None = typer.Option(
        None, help="MovieLens zip. Defaults to the configured one."
    ),
) -> None:
    """Convert the MovieLens archive to parquet and measure what it covers."""
    settings = get_settings()
    settings.ensure_dirs()
    zip_path = archive or settings.movielens_zip

    if not zip_path.exists():
        raise typer.BadParameter(
            f"{zip_path} not found. Download ml-32m.zip from "
            "https://grouplens.org/datasets/movielens/ and drop it in "
            f"{settings.external_dir}."
        )
    if not settings.film_map_path.exists() or not settings.films_tmdb_path.exists():
        raise typer.BadParameter("run `lbrec resolve` and `lbrec enrich` first.")

    console.print(f"preparing {zip_path.name} (first run converts ~836 MB of CSV)...")
    written = prepare(zip_path, settings.movielens_dir)
    console.print("prepared: " + ", ".join(sorted(written)))

    links = load(settings.movielens_dir, "links")
    film_map = pd.read_parquet(settings.film_map_path)
    catalogue = pd.read_parquet(settings.films_tmdb_path)
    catalogue = catalogue[catalogue["media_type"] == "movie"]

    linked = link_films(film_map[film_map["media_type"] == "movie"], links)
    covered = int(linked["in_movielens"].sum())
    console.print(
        f"[bold]{covered}[/bold] / {len(linked)} of our films are in MovieLens "
        f"({covered / len(linked):.1%})"
    )

    ratings = load(settings.movielens_dir, "ratings", columns=["movieId"])
    per_movie = ratings["movieId"].value_counts()
    console.print(
        f"MovieLens holds [bold]{len(ratings):,}[/bold] ratings over "
        f"[bold]{len(per_movie):,}[/bold] rated films"
    )

    report = coverage_by_popularity(linked, catalogue, per_movie)
    _render(report, "MovieLens coverage by catalogue popularity decile")


@app.command()
def profile(
    user: str | None = typer.Option(None, help="Whose profile to report (default: 'me')."),
) -> None:
    """Report this user's taste against catalogue popularity.

    Per-user diagnostics, not project constants: they say which baseline a model
    has to beat for this person and whether their history has a long tail to
    surface at all.
    """
    settings = get_settings()
    paths = settings.user(user)
    for path in (paths.film_status, settings.film_map_path, settings.films_tmdb_path):
        if not path.exists():
            raise typer.BadParameter(f"{path} not found. Run ingest, resolve and enrich first.")

    status = pd.read_parquet(paths.film_status)
    film_map = pd.read_parquet(settings.film_map_path)
    films = pd.read_parquet(settings.films_tmdb_path)

    joined = (
        film_map[film_map["tmdb_id"].notna()][["film_key", "tmdb_id"]]
        .merge(status, on="film_key")
        .merge(films, on="tmdb_id", suffixes=("", "_tmdb"))
    )
    # TV is taste signal only and never a candidate, so it is left out of
    # popularity statistics that describe the recommendable catalogue.
    joined = joined[joined["media_type"] == "movie"]
    rated = joined[joined["rating"].notna()]
    pending = joined[joined["watchlist_pending"]]

    result = build_profile(rated, films[films["media_type"] == "movie"])
    _render(result.summary, f"Taste profile: {paths.user}")
    _render(result.deciles, "Rated films by catalogue popularity decile")
    _render(result.tail, "How obscure this library gets")
    _render(popularity_gap(rated, pending), "Watched versus wanted")


@app.command()
def factors(
    n_factors: int = typer.Option(64, help="Latent dimensions."),
    min_ratings: int = typer.Option(20, help="Skip items with fewer MovieLens ratings."),
) -> None:
    """Learn MovieLens item factors, the collaborative half of the fold-in.

    Computed once from other people's ratings only, so they never see any local
    user's labels and can be reused across folds and across users.
    """
    settings = get_settings()
    settings.ensure_dirs()
    ratings_path = settings.movielens_dir / "ratings.parquet"
    if not ratings_path.exists():
        raise typer.BadParameter("MovieLens not prepared. Run `lbrec movielens` first.")

    console.print("loading 32M ratings...")
    ratings = load(settings.movielens_dir, "ratings", columns=["userId", "movieId", "rating"])
    console.print(f"factorising {len(ratings):,} ratings into {n_factors} dimensions...")
    table = build_item_factors(ratings, n_factors=n_factors, min_item_ratings=min_ratings)
    table.to_parquet(settings.item_factors_path, index=False)

    console.print(
        f"[bold]{len(table):,}[/bold] items with factors "
        f"(>= {min_ratings} ratings), "
        f"explained variance {table.attrs['explained_variance']:.1%}"
    )
    console.print(f"wrote {settings.item_factors_path.relative_to(settings.artifacts_dir.parent)}")


@app.command("models")
def list_models() -> None:
    """List the models `lbrec evaluate --models` accepts."""
    settings = get_settings()
    has_factors = settings.item_factors_path.exists()
    rows = [
        {"model": name, "needs": "" if name in CONTENT_ONLY else "lbrec factors"}
        for name in available_models(with_factors=True)
    ]
    _render(pd.DataFrame(rows), "Available models")
    if not has_factors:
        console.print(
            "[yellow]item factors absent: run `lbrec factors` to enable the "
            "collaborative models[/yellow]"
        )


@app.command("evaluate")
def evaluate_models(
    user: str | None = typer.Option(None, help="Whose ratings to evaluate on (default: 'me')."),
    models: str = typer.Option(
        "", help="Comma-separated model names. Empty means the default set. See `lbrec models`."
    ),
    folds: int = typer.Option(5, help="Cross-validation folds."),
    repeats: int = typer.Option(1, help="Repeat the whole split under fresh seeds for error bars."),
    seed: int = typer.Option(0, help="Base random seed."),
    output: Path | None = typer.Option(None, help="Write per-decile results to this CSV."),
) -> None:
    """Cross-validate models, reporting overall and per popularity decile."""
    settings = get_settings()
    paths = settings.user(user)
    for path in (paths.film_status, settings.film_map_path, settings.films_tmdb_path):
        if not path.exists():
            raise typer.BadParameter(f"{path} not found. Run ingest, resolve and enrich first.")

    status = pd.read_parquet(paths.film_status)
    film_map = pd.read_parquet(settings.film_map_path)
    catalogue = pd.read_parquet(settings.films_tmdb_path)
    catalogue = catalogue[catalogue["media_type"] == "movie"]

    joined = (
        film_map[film_map["tmdb_id"].notna()][["film_key", "tmdb_id"]]
        .merge(status[["film_key", "rating"]], on="film_key")
        .merge(catalogue, on="tmdb_id")
        .dropna(subset=["rating"])
        .drop_duplicates("tmdb_id")
        .reset_index(drop=True)
    )
    if len(joined) < folds * 2:
        raise typer.BadParameter(f"only {len(joined)} rated films with metadata; too few.")

    factor_table = links_table = None
    if settings.item_factors_path.exists() and (settings.movielens_dir / "links.parquet").exists():
        factor_table = pd.read_parquet(settings.item_factors_path)
        links_table = load(settings.movielens_dir, "links")

    if models.strip():
        names = [n.strip() for n in models.split(",") if n.strip()]
        try:
            chosen = [build_model(n, factor_table, links_table) for n in names]
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        chosen = default_models(factor_table, links_table)

    features = build_film_features(joined)
    ratings = joined["rating"].astype(float)
    console.print(
        f"evaluating [bold]{len(chosen)}[/bold] model(s) on [bold]{len(joined)}[/bold] rated films "
        f"for [bold]{paths.user}[/bold]: {folds}-fold x {repeats} repeat(s)"
    )

    with tqdm(total=repeats, desc="repeats", unit="run") as bar:
        result = evaluate(
            chosen,
            features,
            ratings,
            catalogue["vote_count"],
            folds=folds,
            random_state=seed,
            repeats=repeats,
            progress=bar,
        )

    overall = result.overall.copy()
    for column in ("rmse", "rmse_sd", "mae", "spearman"):
        overall[column] = overall[column].astype(float).round(3)
    _render(overall, "Overall (lower RMSE better; rmse_sd is spread across repeats)")
    _render(tail_summary(result.by_decile), "Head vs tail RMSE (tail = deciles 1-3)")

    pivot = (
        result.by_decile.pivot_table(index="decile", columns="model", values="rmse")
        .round(3)
        .reset_index()
    )
    _render(pivot, "RMSE by popularity decile")

    for model in chosen:
        if isinstance(model, StackedEnsemble):
            _render(model.describe_weights(), f"Fitted weights: {model.name}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.by_decile.to_csv(output, index=False)
        console.print(f"wrote {output}")


@app.command("resolve-apply")
def resolve_apply() -> None:
    """Fold the reviewed unresolved.csv into the committed overrides file."""
    settings = get_settings()
    rows = read_reviewed(settings.unresolved_path)
    if not rows:
        console.print(
            f"No filled-in rows found in {settings.unresolved_path}. "
            "Add a tmdb_id (or a note, to record a deliberate non-match) and re-run."
        )
        raise typer.Exit()

    known = load_overrides(settings.overrides_path)
    fresh = [row for row in rows if row["film_key"] not in known]
    written = append_overrides(settings.overrides_path, fresh)

    console.print(
        f"appended [bold]{written}[/bold] override(s) to "
        f"{settings.overrides_path.relative_to(settings.artifacts_dir.parent)}"
        + (f" ({len(rows) - written} already present)" if len(rows) > written else "")
    )
    console.print("re-run `lbrec resolve` to pick them up")


if __name__ == "__main__":
    app()
