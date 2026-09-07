"""Interactive dashboard: taste diagnostics, catalogue stats and a model
hyperparameter playground.

Run with::

    uv run streamlit run src/lbrec/dashboard/app.py

Everything here reads the parquet artifacts the CLI pipeline already writes
(``lbrec ingest/resolve/enrich/movielens/factors``); it computes nothing the
pipeline does not already compute, and re-exposes ``profile``/``evaluate``
under Streamlit's cache and widgets.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run` to be invoked directly against this file without the
# package having been installed into the active environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lbrec.dashboard import data as dd
from lbrec.dashboard.playground_models import ALL_NAMES, NEEDS_FACTORS, build_playground_model
from lbrec.evaluate import evaluate as run_evaluate
from lbrec.evaluate import tail_summary
from lbrec.profile import build_profile, popularity_gap

st.set_page_config(page_title="Letterboxd recommender", page_icon="🎬", layout="wide")

PALETTE = px.colors.qualitative.Set2


def _chart(fig: go.Figure) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.title("🎬 Letterboxd recommender -- diagnostics & model playground")

if not dd.pipeline_ready():
    st.error(dd.CATALOGUE_MISSING)
    st.stop()

users = dd.known_users()
if not users:
    st.error("No ingested export found. Run `lbrec ingest` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    user = st.selectbox("User", users, index=users.index("me") if "me" in users else 0)
    has_factors = dd.factors_ready()
    if not has_factors:
        st.info("`lbrec factors` not run -- collaborative/hybrid models are unavailable.")

rated = dd.load_rated(user)
pending = dd.load_pending(user)
catalogue = dd.load_catalogue()

tab_overview, tab_profile, tab_playground, tab_explore = st.tabs(
    ["Overview", "Taste profile", "Model playground", "Explore predictions"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with tab_overview:
    joined = dd.load_joined(user)
    watched = joined[joined["seen"]] if "seen" in joined else joined

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rated films", len(rated))
    c2.metric("Watched", int(joined["seen"].sum()) if "seen" in joined else len(joined))
    c3.metric("Watchlist pending", len(pending))
    c4.metric("Mean rating", f"{rated['rating'].astype(float).mean():.2f}" if len(rated) else "-")
    c5.metric(
        "Rating sd", f"{rated['rating'].astype(float).std(ddof=0):.2f}" if len(rated) else "-"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Rating distribution")
        counts = rated["rating"].astype(float).value_counts().sort_index().reset_index()
        counts.columns = ["rating", "films"]
        fig = px.bar(counts, x="rating", y="films", color_discrete_sequence=PALETTE)
        _chart(fig)

    with right:
        st.subheader("Popularity: catalogue vs. what you've rated")
        cat_votes = np.log1p(catalogue["vote_count"].astype(float).dropna())
        rated_votes = np.log1p(rated["vote_count"].astype(float).dropna())
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=cat_votes, name="catalogue", histnorm="probability density", opacity=0.6)
        )
        fig.add_trace(
            go.Histogram(x=rated_votes, name="rated", histnorm="probability density", opacity=0.6)
        )
        fig.update_layout(barmode="overlay", xaxis_title="log(1 + vote_count)")
        fig.update_traces(marker_line_width=0)
        _chart(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Rating by genre")
        exploded = dd.explode_list_column(rated, "genres")
        if len(exploded):
            summary = (
                exploded.groupby("genres")
                .agg(films=("rating", "size"), mean_rating=("rating", "mean"))
                .query("films >= 5")
                .sort_values("films", ascending=False)
                .head(15)
                .reset_index()
            )
            fig = px.bar(
                summary.sort_values("mean_rating"),
                x="mean_rating",
                y="genres",
                orientation="h",
                color="mean_rating",
                color_continuous_scale="Tealrose",
                hover_data={"films": True},
            )
            _chart(fig)
        else:
            st.caption("No genre data.")

    with right:
        st.subheader("Rating by decade")
        decade = rated.copy()
        decade["decade"] = (pd.to_numeric(decade["year"], errors="coerce") // 10 * 10).astype(
            "Int64"
        )
        summary = (
            decade.dropna(subset=["decade"])
            .groupby("decade")
            .agg(films=("rating", "size"), mean_rating=("rating", "mean"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=summary["decade"], y=summary["films"], name="films", yaxis="y2", opacity=0.35)
        )
        fig.add_trace(
            go.Scatter(
                x=summary["decade"],
                y=summary["mean_rating"],
                name="mean rating",
                mode="lines+markers",
            )
        )
        fig.update_layout(
            yaxis=dict(title="mean rating"),
            yaxis2=dict(title="films", overlaying="y", side="right", showgrid=False),
            xaxis_title="decade",
        )
        _chart(fig)

    st.subheader("Most-watched directors")
    exploded_dir = dd.explode_list_column(rated, "directors")
    if len(exploded_dir):
        summary = (
            exploded_dir.groupby("directors")
            .agg(films=("rating", "size"), mean_rating=("rating", "mean"))
            .query("films >= 3")
            .sort_values("films", ascending=False)
            .head(20)
            .reset_index()
        )
        fig = px.scatter(
            summary,
            x="films",
            y="mean_rating",
            text="directors",
            size="films",
            color="mean_rating",
            color_continuous_scale="Tealrose",
        )
        fig.update_traces(textposition="top center")
        _chart(fig)
    else:
        st.caption("No director data.")

# --------------------------------------------------------------------------
# Taste profile
# --------------------------------------------------------------------------

with tab_profile:
    if not len(rated):
        st.warning("No rated films for this user.")
    else:
        result = build_profile(rated, catalogue)
        summary = result.summary.set_index("metric")["value"]

        c1, c2, c3 = st.columns(3)
        c1.metric("rho(rating, crowd score)", summary.get("rho(rating, crowd score)", "-"))
        c1.caption(
            "THE baseline to beat -- if a model can't clear this it learned nothing personal."
        )
        c2.metric("rho(rating, log vote_count)", summary.get("rho(rating, log vote_count)", "-"))
        c2.caption("Near zero means popularity doesn't already explain this taste.")
        c3.metric("RMSE floor (predict the mean)", summary.get("RMSE floor", "-"))

        st.subheader("Rated films by catalogue popularity decile")
        st.caption(
            "Deciles are cut against the whole catalogue, so decile 1 means the same "
            "thing for everyone."
        )
        deciles = result.deciles
        if len(deciles):
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=deciles["decile"], y=deciles["films"], name="films", yaxis="y2", opacity=0.35
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=deciles["decile"],
                    y=deciles["mean_rating"],
                    name="mean rating",
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                yaxis=dict(title="mean rating"),
                yaxis2=dict(title="films", overlaying="y", side="right", showgrid=False),
                xaxis=dict(title="popularity decile (1 = most obscure)", dtick=1),
            )
            _chart(fig)

        left, right = st.columns(2)
        with left:
            st.subheader("How obscure this library gets")
            tail = result.tail
            fig = px.bar(
                tail, x="under_votes", y="films", text="share", color_discrete_sequence=PALETTE
            )
            fig.update_xaxes(type="category", title="vote_count under")
            _chart(fig)

        with right:
            st.subheader("Watched vs. wanted (popularity)")
            gap = popularity_gap(rated, pending)
            fig = go.Figure()
            for band, color in zip(
                ("p10_votes", "median_votes", "p90_votes"), PALETTE, strict=False
            ):
                fig.add_trace(go.Bar(x=gap["set"], y=gap[band], name=band, marker_color=color))
            fig.update_layout(yaxis_title="vote_count", barmode="group")
            _chart(fig)
            st.caption(
                "If the watchlist skews far more popular than what you've rated, "
                "it's availability-biased and less safe to use as a held-out positive set."
            )

# --------------------------------------------------------------------------
# Model playground
# --------------------------------------------------------------------------

with tab_playground:
    st.caption(
        "Cross-validates the chosen models on your rated films, sliced by popularity decile -- "
        "exactly what `lbrec evaluate` reports, but with hyperparameters you can move live."
    )

    available = [n for n in ALL_NAMES if has_factors or n not in NEEDS_FACTORS]
    default_selection = [
        n for n in ("global_mean", "crowd_score", "content_ridge") if n in available
    ]
    chosen_names = st.multiselect("Models", available, default=default_selection)

    col_a, col_b, col_c = st.columns(3)
    folds = col_a.slider("CV folds", min_value=3, max_value=10, value=5)
    repeats = col_b.slider("Repeats (error bars)", min_value=1, max_value=10, value=1)
    seed = col_c.number_input("Seed", min_value=0, value=0, step=1)

    params: dict[str, dict] = {}
    needs_text_params = {
        "content_ridge",
        "content_ridge_debiased",
        "content_knn",
        "gradient_boosted",
        "hybrid",
    }
    for name in chosen_names:
        if name not in (needs_text_params | {"gradient_boosted", "collaborative_fold", "hybrid"}):
            continue
        with st.expander(f"Hyperparameters -- {name}"):
            p: dict = {}
            if name in needs_text_params:
                p["keyword_components"] = st.slider(
                    "keyword_components (SVD dims)", 8, 200, 96, key=f"{name}_kw"
                )
                p["synopsis_components"] = st.slider(
                    "synopsis_components (SVD dims)", 8, 150, 64, key=f"{name}_syn"
                )
            if name == "content_knn":
                p["n_neighbors"] = st.slider("n_neighbors", 3, 60, 25, key=f"{name}_k")
            if name == "gradient_boosted":
                p["n_estimators"] = st.slider(
                    "n_estimators", 50, 1000, 400, step=50, key=f"{name}_ne"
                )
                p["max_depth"] = st.slider("max_depth", 1, 10, 4, key=f"{name}_md")
                p["learning_rate"] = st.slider(
                    "learning_rate", 0.01, 0.5, 0.05, step=0.01, key=f"{name}_lr"
                )
                p["subsample"] = st.slider("subsample", 0.3, 1.0, 0.8, step=0.05, key=f"{name}_ss")
                p["colsample_bytree"] = st.slider(
                    "colsample_bytree", 0.3, 1.0, 0.6, step=0.05, key=f"{name}_cs"
                )
                p["min_child_weight"] = st.slider("min_child_weight", 1, 20, 5, key=f"{name}_mcw")
                p["reg_lambda"] = st.slider(
                    "reg_lambda", 0.0, 10.0, 2.0, step=0.5, key=f"{name}_rl"
                )
            if name == "hybrid":
                p["inner_folds"] = st.slider(
                    "inner_folds (blend weight fit)", 3, 8, 4, key=f"{name}_if"
                )
            params[name] = p

    run = st.button("Run cross-validation", type="primary", disabled=not chosen_names)

    if run:
        features, ratings, catalogue_votes = dd.build_features_for(user)
        factors = dd.load_factors() if has_factors else None
        links = dd.load_links() if has_factors else None
        try:
            models = [
                build_playground_model(name, params.get(name, {}), factors, links)
                for name in chosen_names
            ]
        except ValueError as exc:
            st.error(str(exc))
            models = []

        if models:
            with st.spinner(f"Cross-validating {len(models)} model(s)..."):
                result = run_evaluate(
                    models,
                    features,
                    ratings,
                    catalogue_votes,
                    folds=folds,
                    random_state=int(seed),
                    repeats=repeats,
                )
            st.session_state["playground_result"] = result
            st.session_state["playground_features"] = features
            st.session_state["playground_rated"] = (
                dd.load_rated(user).drop_duplicates("tmdb_id").reset_index(drop=True)
            )

    result = st.session_state.get("playground_result")
    if result is not None:
        overall = result.overall.copy()
        for col in ("rmse", "rmse_sd", "mae", "spearman"):
            overall[col] = overall[col].astype(float)

        left, right = st.columns(2)
        with left:
            st.subheader("Overall RMSE (lower is better)")
            fig = px.bar(
                overall.sort_values("rmse"),
                x="model",
                y="rmse",
                error_y="rmse_sd",
                color="model",
                color_discrete_sequence=PALETTE,
            )
            _chart(fig)
        with right:
            st.subheader("Overall Spearman (rank correlation, higher is better)")
            fig = px.bar(
                overall.sort_values("spearman", ascending=False),
                x="model",
                y="spearman",
                color="model",
                color_discrete_sequence=PALETTE,
            )
            _chart(fig)

        st.subheader("RMSE by popularity decile")
        st.caption("The gap between decile 1 and decile 10 is the number that matters most here.")
        by_decile = result.by_decile.dropna(subset=["rmse"])
        fig = px.line(
            by_decile.sort_values("decile"),
            x="decile",
            y="rmse",
            color="model",
            markers=True,
            color_discrete_sequence=PALETTE,
        )
        fig.update_xaxes(dtick=1, title="popularity decile (1 = most obscure)")
        _chart(fig)

        st.subheader("Head vs. tail RMSE (tail = deciles 1-3)")
        st.dataframe(tail_summary(result.by_decile), width="stretch", hide_index=True)

        for model in models if run and models else []:
            if hasattr(model, "describe_weights"):
                st.subheader(f"Fitted weights -- {model.name}")
                st.dataframe(model.describe_weights(), width="stretch", hide_index=True)
            elif hasattr(model, "weight_"):
                st.caption(f"{model.name}: fitted collaborative-leg weight = {model.weight_:.3f}")
    else:
        st.info("Choose models and click **Run cross-validation**.")

# --------------------------------------------------------------------------
# Explore predictions
# --------------------------------------------------------------------------

with tab_explore:
    result = st.session_state.get("playground_result")
    if result is None:
        st.info("Run the model playground first -- predictions from that run show up here.")
    else:
        predictions = result.predictions
        rated_for_titles = st.session_state["playground_rated"].set_index(
            st.session_state["playground_features"].index
        )
        model_options = sorted(predictions["model"].unique())
        picked = st.selectbox("Model", model_options)

        subset = predictions[predictions["model"] == picked].copy()
        subset = subset.join(
            rated_for_titles[["title", "year", "vote_count"]], on="row", how="left"
        )
        subset["abs_error"] = (subset["actual"] - subset["predicted"]).abs()
        subset["decile"] = pd.to_numeric(subset["decile"], errors="coerce")

        st.subheader(f"Actual vs. predicted -- {picked}")
        fig = px.scatter(
            subset,
            x="actual",
            y="predicted",
            color="decile",
            color_continuous_scale="Tealrose",
            hover_data={"title": True, "year": True, "abs_error": ":.2f"},
        )
        lo, hi = subset["actual"].min(), subset["actual"].max()
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(dash="dash"))
        )
        _chart(fig)

        st.subheader("Largest misses")
        top_errors = subset.sort_values("abs_error", ascending=False).head(25)
        st.dataframe(
            top_errors[
                ["title", "year", "vote_count", "decile", "actual", "predicted", "abs_error"]
            ].round({"actual": 2, "predicted": 2, "abs_error": 2}),
            width="stretch",
            hide_index=True,
        )
