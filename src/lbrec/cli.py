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
from lbrec.letterboxd import coverage_report, film_status, load_export
from lbrec.resolve import (
    REVIEW_CONFIDENCES,
    TRUSTED_CONFIDENCES,
    append_overrides,
    build_review_table,
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
    export_dir: Path | None = typer.Option(
        None, help="Letterboxd export directory. Defaults to the configured export dir."
    ),
    include_deleted: bool = typer.Option(
        True, help="Include deleted/ and orphaned/ entries, flagged instead of dropped."
    ),
) -> None:
    """Parse the Letterboxd export into films_local.parquet and interactions.parquet."""
    settings = get_settings()
    settings.ensure_dirs()
    source = export_dir or settings.export_dir

    if not source.exists():
        raise typer.BadParameter(f"Export directory not found: {source}")

    export = load_export(source, include_deleted=include_deleted)
    if export.interactions.empty:
        console.print(f"[red]No interactions parsed from {source}.[/red]")
        raise typer.Exit(code=1)

    status = film_status(export.interactions)
    export.films_local.to_parquet(settings.films_local_path, index=False)
    export.interactions.to_parquet(settings.interactions_path, index=False)
    status.to_parquet(settings.film_status_path, index=False)

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
    for path in (settings.films_local_path, settings.interactions_path, settings.film_status_path):
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

    if not settings.films_local_path.exists():
        raise typer.BadParameter("films_local.parquet not found -- run `lbrec ingest` first.")

    films = pd.read_parquet(settings.films_local_path)[["film_key", "title", "year"]]
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
    pending = int(film_map["confidence"].isin(REVIEW_CONFIDENCES).sum())
    console.print(
        f"[bold]{trusted}[/bold] / {len(film_map)} usable without review "
        f"([bold]{pending}[/bold] awaiting a human)"
    )
    console.print(f"wrote {settings.film_map_path.relative_to(settings.artifacts_dir.parent)}")
    if review and pending:
        console.print(
            f"wrote {settings.unresolved_path.relative_to(settings.artifacts_dir.parent)} "
            "-- fill in the tmdb_id column, then run `lbrec resolve-apply`"
        )


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
