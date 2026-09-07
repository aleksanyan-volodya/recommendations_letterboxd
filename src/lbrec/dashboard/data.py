"""Cached data loading and joining for the dashboard.

Every function here is a thin, cached wrapper over the parquet artifacts the
pipeline already writes (``lbrec ingest/resolve/enrich/movielens/factors``).
Nothing here computes anything the pipeline does not already compute -- it
only joins tables for display and re-exposes ``profile``/``evaluate`` under
Streamlit's cache so the dashboard stays responsive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lbrec.config import get_settings
from lbrec.features import build_film_features
from lbrec.movielens import load as load_movielens

CATALOGUE_MISSING = (
    "Missing artifacts. Run the pipeline first: "
    "`lbrec ingest`, `lbrec resolve`, `lbrec enrich` (and `lbrec movielens` + "
    "`lbrec factors` for the collaborative models)."
)


@st.cache_resource
def settings():
    return get_settings()


@st.cache_data
def known_users() -> list[str]:
    return settings().known_users()


@st.cache_data
def pipeline_ready() -> bool:
    s = settings()
    return s.film_map_path.exists() and s.films_tmdb_path.exists()


@st.cache_data
def load_catalogue() -> pd.DataFrame:
    """Every resolved title, movies only -- the recommendable catalogue."""
    films = pd.read_parquet(settings().films_tmdb_path)
    return films[films["media_type"] == "movie"].reset_index(drop=True)


@st.cache_data
def load_film_map() -> pd.DataFrame:
    return pd.read_parquet(settings().film_map_path)


@st.cache_data
def load_status(user: str) -> pd.DataFrame:
    return pd.read_parquet(settings().user(user).film_status)


@st.cache_data
def load_interactions(user: str) -> pd.DataFrame:
    return pd.read_parquet(settings().user(user).interactions)


@st.cache_data
def load_joined(user: str) -> pd.DataFrame:
    """film_status + film_map + films_tmdb, movies only, one row per film."""
    status = load_status(user)
    film_map = load_film_map()
    catalogue = pd.read_parquet(settings().films_tmdb_path)
    joined = (
        film_map[film_map["tmdb_id"].notna()][["film_key", "tmdb_id"]]
        .merge(status, on="film_key")
        .merge(catalogue, on="tmdb_id", suffixes=("", "_tmdb"))
    )
    return joined[joined["media_type"] == "movie"].reset_index(drop=True)


@st.cache_data
def load_rated(user: str) -> pd.DataFrame:
    joined = load_joined(user)
    return joined[joined["rating"].notna()].reset_index(drop=True)


@st.cache_data
def load_pending(user: str) -> pd.DataFrame:
    joined = load_joined(user)
    return joined[joined["watchlist_pending"]].reset_index(drop=True)


@st.cache_data
def factors_ready() -> bool:
    s = settings()
    return s.item_factors_path.exists() and (s.movielens_dir / "links.parquet").exists()


@st.cache_data
def load_factors() -> pd.DataFrame:
    return pd.read_parquet(settings().item_factors_path)


@st.cache_data
def load_links() -> pd.DataFrame:
    return load_movielens(settings().movielens_dir, "links")


@st.cache_data
def explode_list_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Long-format explode of a list column (genres, directors, cast, ...).

    Handles the parquet-roundtrip quirk where list columns come back as numpy
    arrays rather than Python lists.
    """
    frame = df[[column, "rating"]].copy()
    frame[column] = frame[column].map(
        lambda v: list(v) if isinstance(v, (list, np.ndarray)) else []
    )
    return frame.explode(column).dropna(subset=[column])


@st.cache_data
def build_features_for(user: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix, ratings and catalogue votes for evaluation, for one user."""
    rated = load_rated(user).drop_duplicates("tmdb_id").reset_index(drop=True)
    catalogue = load_catalogue()
    features = build_film_features(rated)
    ratings = rated["rating"].astype(float)
    return features, ratings, catalogue["vote_count"]
