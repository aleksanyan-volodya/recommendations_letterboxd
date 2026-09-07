"""Build models from playground hyperparameters.

Separate from ``lbrec.models.build_model`` because that factory only ever
constructs the fixed configuration ``lbrec evaluate`` uses. The dashboard
needs the same model classes constructed with whatever values the sliders are
currently set to, so this module is a thin, UI-facing alternative factory
rather than a change to the pipeline's own defaults.
"""

from __future__ import annotations

import pandas as pd

from lbrec.models import CollaborativeFold as _CollaborativeFold
from lbrec.models import (
    ContentKNN,
    ContentRidge,
    CrowdScore,
    GlobalMean,
    GradientBoosted,
    Model,
    StackedEnsemble,
)
from lbrec.models import SupportGatedBlend as _SupportGatedBlend

#: model name -> whether it needs MovieLens factors/links.
NEEDS_FACTORS = {"collaborative_fold", "hybrid"}

ALL_NAMES = [
    "global_mean",
    "crowd_score",
    "content_ridge",
    "content_ridge_debiased",
    "content_knn",
    "gradient_boosted",
    "collaborative_fold",
    "hybrid",
]


def build_playground_model(
    name: str,
    params: dict,
    factors: pd.DataFrame | None,
    links: pd.DataFrame | None,
) -> Model:
    if name in NEEDS_FACTORS and (factors is None or links is None):
        raise ValueError(f"{name!r} needs MovieLens item factors -- run `lbrec factors` first")

    if name == "global_mean":
        return GlobalMean()
    if name == "crowd_score":
        return CrowdScore()
    if name == "content_ridge":
        return ContentRidge(
            keyword_components=params["keyword_components"],
            synopsis_components=params["synopsis_components"],
        )
    if name == "content_ridge_debiased":
        return ContentRidge(
            debias=True,
            keyword_components=params["keyword_components"],
            synopsis_components=params["synopsis_components"],
        )
    if name == "content_knn":
        return ContentKNN(
            n_neighbors=params["n_neighbors"],
            keyword_components=params["keyword_components"],
            synopsis_components=params["synopsis_components"],
        )
    if name == "gradient_boosted":
        return GradientBoosted(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            keyword_components=params["keyword_components"],
            synopsis_components=params["synopsis_components"],
        )
    if name == "collaborative_fold":
        return _CollaborativeFold(factors, links)
    if name == "hybrid":
        content = ContentRidge(
            keyword_components=params["keyword_components"],
            synopsis_components=params["synopsis_components"],
        )
        collab = _CollaborativeFold(factors, links)
        return _SupportGatedBlend(content, collab, inner_folds=params["inner_folds"])
    raise ValueError(f"unknown model {name!r}")


def maybe_stacked(models: list[Model], *, name: str = "ensemble") -> Model:
    """Wrap more than one model in an out-of-fold stacked ensemble."""
    return StackedEnsemble(models, name=name)
