"""Load a MovieLens archive and measure what it covers.

MovieLens is the substitute for the user base this project does not have: 32M
ratings from 200k people, which is where any collaborative signal has to come
from when there is only one real user. It is used for research only -- free
redistribution under the same licence is allowed, commercial use is not.

The archive is read straight from the downloaded zip and converted to parquet
once. Nothing here mutates the zip, so the whole pipeline stays reproducible
from the original download.

The join runs through ``links.csv``, and it has a trap: **MovieLens stores IMDb
ids as bare integers with the ``tt`` prefix and leading zeros stripped**, so
``114709`` means ``tt0114709``. Joining on the raw column silently matches
nothing.

The measurement that matters here is not "how many of our films does MovieLens
know" but "how does that coverage vary with popularity". MovieLens is itself a
popularity-biased sample, so if its coverage collapses in the low deciles then
the collaborative leg cannot serve the long tail and the content leg has to.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

#: Members worth converting, with explicit types. Left to its own devices
#: pyarrow widens these and the ratings table costs several times more memory.
MEMBERS: dict[str, dict[str, pa.DataType]] = {
    "ratings": {
        "userId": pa.int32(),
        "movieId": pa.int32(),
        "rating": pa.float32(),
        "timestamp": pa.int64(),
    },
    "movies": {"movieId": pa.int32(), "title": pa.string(), "genres": pa.string()},
    "links": {"movieId": pa.int32(), "imdbId": pa.int64(), "tmdbId": pa.float64()},
    "tags": {
        "userId": pa.int32(),
        "movieId": pa.int32(),
        "tag": pa.string(),
        "timestamp": pa.int64(),
    },
}


def imdb_tt(numeric: pd.Series) -> pd.Series:
    """Turn MovieLens' bare IMDb integers into ``tt``-prefixed ids.

    ``114709`` -> ``tt0114709``. Ids are zero-padded to seven digits, but longer
    ones are left as they are rather than truncated.
    """
    numbers = pd.to_numeric(numeric, errors="coerce").astype("Int64")
    formatted = numbers.map(lambda value: f"tt{int(value):07d}" if pd.notna(value) else pd.NA)
    return formatted.astype("string")


def archive_prefix(archive: zipfile.ZipFile) -> str:
    """The single top-level directory inside a MovieLens zip (``ml-32m/``)."""
    roots = {name.split("/")[0] for name in archive.namelist() if "/" in name}
    if len(roots) != 1:
        raise ValueError(f"expected one top-level directory in the archive, found {sorted(roots)}")
    return next(iter(roots))


def prepare(zip_path: Path, out_dir: Path, *, members: list[str] | None = None) -> dict[str, Path]:
    """Convert archive members to parquet, skipping any already converted.

    Each member is extracted to a temporary file before parsing rather than read
    through the zip stream: ``ratings.csv`` is 836 MB uncompressed and pyarrow is
    markedly faster against a real file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = members or list(MEMBERS)
    written: dict[str, Path] = {}

    with zipfile.ZipFile(zip_path) as archive:
        prefix = archive_prefix(archive)
        available = set(archive.namelist())

        for name in wanted:
            target = out_dir / f"{name}.parquet"
            source = f"{prefix}/{name}.csv"
            if target.exists():
                written[name] = target
                continue
            if source not in available:
                continue  # e.g. genome files, absent from ml-32m

            with tempfile.TemporaryDirectory() as scratch:
                extracted = Path(scratch) / f"{name}.csv"
                with archive.open(source) as reader, extracted.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=8 << 20)
                table = pa_csv.read_csv(
                    extracted,
                    convert_options=pa_csv.ConvertOptions(column_types=MEMBERS[name]),
                )
                pq.write_table(table, target, compression="zstd")
            written[name] = target

    return written


def load(out_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one prepared table."""
    return pq.read_table(out_dir / f"{name}.parquet", columns=columns).to_pandas()


def link_films(film_map: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Attach MovieLens ``movieId`` to our films, joining on TMDb id.

    TMDb is the primary key rather than IMDb because our resolution step
    produces it directly, and ``links.csv`` carries both.
    """
    ml = links.copy()
    ml["tmdbId"] = pd.to_numeric(ml["tmdbId"], errors="coerce").astype("Int64")
    ml = ml.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")

    films = film_map[film_map["tmdb_id"].notna()].copy()
    films["tmdb_id"] = films["tmdb_id"].astype("Int64")
    joined = films.merge(
        ml[["movieId", "tmdbId", "imdbId"]], left_on="tmdb_id", right_on="tmdbId", how="left"
    )
    joined["in_movielens"] = joined["movieId"].notna()
    return joined


def _bridge(links: pd.DataFrame) -> pd.DataFrame:
    """``links.csv`` made one-to-one: blank and repeated tmdbIds would fan out."""
    bridge = links[["movieId", "tmdbId"]].copy()
    bridge["tmdbId"] = pd.to_numeric(bridge["tmdbId"], errors="coerce")
    bridge = bridge.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId").drop_duplicates("movieId")
    bridge["tmdbId"] = bridge["tmdbId"].astype("int64")
    return bridge


def _rekey(values: pd.Series, source: pd.Series, target: pd.Series, name: str) -> pd.Series:
    mapped = source.map(values)
    keep = mapped.notna().to_numpy()
    index = pd.Index(target[keep].to_numpy(), name=name)
    return pd.Series(mapped[keep].to_numpy(), index=index, name=values.name)


def by_movie_id(by_tmdb: pd.Series, links: pd.DataFrame) -> pd.Series:
    """Re-key a TMDb-indexed series to MovieLens ``movieId``, one to one."""
    bridge = _bridge(links)
    return _rekey(by_tmdb, bridge["tmdbId"], bridge["movieId"], "movieId")


def by_tmdb_id(by_movie: pd.Series, links: pd.DataFrame) -> pd.Series:
    """Re-key a movieId-indexed series to TMDb id, one to one."""
    bridge = _bridge(links)
    return _rekey(by_movie, bridge["movieId"], bridge["tmdbId"], "tmdb_id")


def item_bias(ratings: pd.DataFrame, *, prior: float = 20.0) -> pd.Series:
    """Each film's shrunk departure from the MovieLens mean, by ``movieId``.

    The models' own estimator, ``(sum - n * mean) / (n + prior)``, so it can be
    lent to a model fitted on another crowd and mean the same thing.
    """
    mean = ratings["rating"].mean()
    grouped = ratings.groupby("movieId")["rating"].agg(["sum", "count"])
    return ((grouped["sum"] - grouped["count"] * mean) / (grouped["count"] + prior)).rename("bias")


def rated_tmdb_ids(links: pd.DataFrame, rated_movie_ids) -> set[int]:
    """TMDb ids of films with at least one MovieLens rating.

    Not the same as every TMDb id in ``links.csv``: ml-32m links 3,150 films
    that nobody in it rated. They have no item bias, so for anything that asks
    "does the crowd know this film" they are as cold as a film MovieLens has
    never heard of. Counting them as known understated the pool's crowd-less
    share by over a point.
    """
    rated = links[links["movieId"].isin(set(rated_movie_ids))]
    return set(pd.to_numeric(rated["tmdbId"], errors="coerce").dropna().astype("int64"))


# Items rated fewer times than this get no factor. A handful of ratings
# produces a direction indistinguishable from noise.
MIN_ITEM_RATINGS = 20

# Latent dimensions. Small as the user vector is solved from a few
# hundred personal ratings, so a wide factor space would overfit.
N_FACTORS = 64


def build_item_factors(
    ratings: pd.DataFrame,
    *,
    n_factors: int = N_FACTORS,
    min_item_ratings: int = MIN_ITEM_RATINGS,
    random_state: int = 0,
) -> pd.DataFrame:
    """Latent item vectors from the MovieLens rating matrix.

    This is the collaborative half of the fold-in. Item factors are learned from
    other people's ratings only, so they can be computed once and reused across
    folds and users without leaking. What happens per fold is only the cheap
    part: solving for the user's own position in this space.

    Ratings are centred on each item's mean before factorisation. Without that,
    the leading components encode how *often* a film is rated rather than how it
    is rated, and popularity would be baked into the representation this project
    is trying to keep free of it.
    """
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    counts = ratings["movieId"].value_counts()
    keep = counts[counts >= min_item_ratings].index
    subset = ratings[ratings["movieId"].isin(keep)]

    movie_codes, movie_ids = pd.factorize(subset["movieId"], sort=True)
    user_codes, _ = pd.factorize(subset["userId"], sort=False)

    values = subset["rating"].astype("float32").to_numpy()
    item_means = np.zeros(len(movie_ids), dtype="float32")
    np.add.at(item_means, movie_codes, values)
    item_means /= np.bincount(movie_codes, minlength=len(movie_ids)).astype("float32")
    centred = values - item_means[movie_codes]

    matrix = csr_matrix(
        (centred, (user_codes, movie_codes)),
        shape=(user_codes.max() + 1, len(movie_ids)),
        dtype="float32",
    )
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    svd.fit(matrix)
    factors = svd.components_.T.astype("float32")  # (items, factors)

    frame = pd.DataFrame(factors, columns=[f"f{i}" for i in range(n_factors)])
    frame.insert(0, "movieId", pd.Series(movie_ids, dtype="int32"))
    frame.insert(1, "item_mean", item_means)
    frame.insert(2, "n_ratings", counts.reindex(movie_ids).to_numpy())
    frame.attrs["explained_variance"] = float(svd.explained_variance_ratio_.sum())
    return frame


def coverage_by_popularity(
    linked: pd.DataFrame,
    catalogue: pd.DataFrame,
    ratings_per_movie: pd.Series | None = None,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """MovieLens coverage of our films across catalogue popularity deciles.

    The headline coverage number hides the thing that matters. MovieLens is a
    popularity-biased sample, so coverage in the top decile says nothing about
    whether the collaborative leg can reach the tail. If it falls away in the
    low deciles, obscure films have to be served by content features instead.
    """
    merged = linked.merge(
        catalogue[["tmdb_id", "vote_count"]].astype({"tmdb_id": "Int64"}), on="tmdb_id", how="left"
    )
    votes = merged["vote_count"].astype("Float64").astype(float)
    edges = np.unique(
        np.quantile(catalogue["vote_count"].dropna().astype(float), np.linspace(0, 1, bins + 1))
    )
    if len(edges) < 3:
        return pd.DataFrame()

    merged["decile"] = pd.cut(votes, bins=edges, labels=range(1, len(edges)), include_lowest=True)
    aggregations = {
        "films": ("in_movielens", "size"),
        "in_movielens": ("in_movielens", "sum"),
    }
    if ratings_per_movie is not None:
        merged["ml_ratings"] = merged["movieId"].map(ratings_per_movie)
        aggregations["median_ml_ratings"] = ("ml_ratings", "median")

    report = (
        merged.groupby("decile", observed=True).agg(**aggregations).reset_index()  # type: ignore[arg-type]
    )
    report["coverage"] = (report["in_movielens"] / report["films"]).map("{:.1%}".format)
    if ratings_per_movie is not None:
        report["median_ml_ratings"] = report["median_ml_ratings"].fillna(0).astype(int)
    return report
