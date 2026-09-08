"""Cross-validated evaluation, always sliced by popularity.

Standard recommender metrics assume many users. With one user the only honest
offline measurement is k-fold cross-validation over that user's own ratings, and
the only chronology available is a logging date, so a temporal split would
measure the order things were entered rather than taste over time. Folds are
therefore random and seeded.

**Every metric is reported per popularity decile as well as overall.** A model
at RMSE 0.72 overall and 0.95 on the bottom two deciles is failing at exactly
the thing this project exists for, and the headline number hides it. This slice
does more for the goal than any individual debiasing technique.

Metrics reported per fold and averaged:

RMSE / MAE
    Error on the user's rating scale.
Spearman
    Rank correlation, which is what a ranked list of recommendations actually
    depends on. A model can have mediocre RMSE and still order films well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lbrec.catalogue import TAIL_BANDS, popularity_band
from lbrec.models import Model
from lbrec.profile import spearman

DEFAULT_FOLDS = 5
RANDOM_STATE = 0


@dataclass(frozen=True)
class Evaluation:
    overall: pd.DataFrame
    by_decile: pd.DataFrame
    predictions: pd.DataFrame


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | None]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return {"n": 0, "rmse": None, "mae": None, "spearman": None}
    error = frame["actual"].astype(float) - frame["predicted"].astype(float)
    return {
        "n": len(frame),
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(error.abs().mean()),
        "spearman": spearman(frame["actual"], frame["predicted"]),
    }


def cross_validate(
    models: list[Model],
    features: pd.DataFrame,
    ratings: pd.Series,
    *,
    folds: int = DEFAULT_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Out-of-fold predictions for every model.

    Each model is refitted from scratch on each training fold, so any
    vectoriser, scaler or rescaling constant is learned without seeing the
    held-out films.
    """
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    records = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features)):
        train_x = features.iloc[train_idx]
        train_y = ratings.iloc[train_idx]
        test_x = features.iloc[test_idx]

        for model in models:
            fitted = model.fit(train_x, train_y)
            predicted = fitted.predict(test_x)
            records.append(
                pd.DataFrame(
                    {
                        "model": model.name,
                        "fold": fold,
                        "row": test_x.index,
                        "actual": ratings.iloc[test_idx].to_numpy(),
                        "predicted": np.asarray(predicted, dtype=float),
                    }
                )
            )

    return pd.concat(records, ignore_index=True)


def evaluate(
    models: list[Model],
    features: pd.DataFrame,
    ratings: pd.Series,
    catalogue_votes: pd.Series,
    *,
    folds: int = DEFAULT_FOLDS,
    random_state: int = RANDOM_STATE,
    repeats: int = 1,
    progress=None,
) -> Evaluation:
    """Run cross-validation and summarise overall and per popularity decile.

    With ``repeats > 1`` the whole split is redone under fresh seeds and the
    spread across repeats is reported alongside the mean. That spread is what
    makes a comparison honest: on a few hundred ratings, two models differing by
    0.01 RMSE on one split may simply have been lucky, and without an error bar
    there is no way to tell an improvement from a reshuffle.
    """
    frames = []
    for repeat in range(repeats):
        run = cross_validate(
            models, features, ratings, folds=folds, random_state=random_state + repeat
        )
        run["repeat"] = repeat
        frames.append(run)
        if progress is not None:
            progress.update(1)
    predictions = pd.concat(frames, ignore_index=True)

    # Absolute popularity bands, not quantiles of whatever set happens to be
    # passed in -- see catalogue.POPULARITY_BANDS.
    bands = popularity_band(features["log_votes"].map(np.expm1))
    predictions["band"] = predictions["row"].map(bands)

    per_repeat = (
        predictions.groupby(["model", "repeat"], sort=False)
        .apply(lambda g: pd.Series(_metrics(g["actual"], g["predicted"])), include_groups=False)
        .reset_index()
    )
    overall = (
        per_repeat.groupby("model", sort=False)
        .agg(
            n=("n", "max"),
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae=("mae", "mean"),
            spearman=("spearman", "mean"),
        )
        .reset_index()
        .sort_values("rmse", ignore_index=True)
    )
    overall["rmse_sd"] = overall["rmse_sd"].fillna(0.0)

    by_decile = (
        predictions.dropna(subset=["band"])
        .groupby(["model", "band"], sort=False, observed=True)
        .apply(lambda g: pd.Series(_metrics(g["actual"], g["predicted"])), include_groups=False)
        .reset_index()
    )
    return Evaluation(overall=overall, by_decile=by_decile, predictions=predictions)


def tail_summary(
    by_decile: pd.DataFrame, *, tail_bands: tuple[str, ...] = TAIL_BANDS
) -> pd.DataFrame:
    """Head-versus-tail RMSE per model, and the gap between them.

    The gap is the number to watch. A model that is only good on famous films
    has not solved this problem, however strong its overall score.
    """
    frame = by_decile.dropna(subset=["rmse"]).copy()
    frame["band"] = np.where(frame["band"].isin(tail_bands), "tail", "head")
    pivot = (
        frame.pivot_table(index="model", columns="band", values="rmse", aggfunc="mean")
        .reset_index()
        .rename(columns={"head": "rmse_head", "tail": "rmse_tail"})
    )
    if {"rmse_head", "rmse_tail"} <= set(pivot.columns):
        pivot["gap"] = (pivot["rmse_tail"] - pivot["rmse_head"]).round(3)
        pivot["rmse_head"] = pivot["rmse_head"].round(3)
        pivot["rmse_tail"] = pivot["rmse_tail"].round(3)
    return pivot.sort_values("rmse_tail", ignore_index=True)
