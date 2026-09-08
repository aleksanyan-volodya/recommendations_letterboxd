"""Rating models, all behind one interface.

Every model implements ``fit(features, ratings)`` and ``predict(features)``, so
swapping a ridge for a gradient booster, a matrix factorisation or an LLM
reranker later is a substitution rather than a rewrite. That interface is the
one architectural commitment worth making early; the model behind it is cheap.

The two baselines are not filler. ``GlobalMean`` is the floor, and
``CrowdScore`` is the bar that actually matters: it predicts what everyone else
thought, rescaled to this user. A personal model that cannot beat it has learned
nothing personal, which is the most common way a recommender flatters itself.
"""

from __future__ import annotations

import copy
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from lbrec.features import (
    FULL_BLOCKS,
    KEYWORD_COMPONENTS,
    SYNOPSIS_COMPONENTS,
    build_preprocessor,
    neutralise_popularity,
)


class Model(Protocol):
    """What every scorer must provide."""

    name: str

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> Model: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class GlobalMean:
    """Predict the user's mean rating for everything. The floor."""

    name = "global_mean"

    def __init__(self) -> None:
        self.mean_ = 3.0

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> GlobalMean:
        self.mean_ = float(ratings.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean_)


class CrowdScore:
    """Predict TMDb's average rating, linearly rescaled onto this user's scale.

    The rescaling is fitted rather than assumed: a 0-10 crowd score does not map
    onto a 1-5 personal scale by simple division, because users differ in both
    generosity and spread. Films without a crowd score fall back to the mean.

    Uses the *shrunk* crowd score: the raw average is meaningless for a film
    with one vote, and this baseline would otherwise inherit that pathology.

    **This is the baseline to beat.** It uses no personal information beyond two
    calibration constants.
    """

    name = "crowd_score"

    def __init__(self) -> None:
        self.slope_ = 0.0
        self.intercept_ = 3.0

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> CrowdScore:
        crowd = pd.to_numeric(features["crowd_shrunk"], errors="coerce").astype(float)
        usable = crowd.notna() & (crowd > 0) & ratings.notna()
        self.intercept_ = float(ratings.mean())
        if usable.sum() >= 3 and crowd[usable].std() > 0:
            self.slope_, self.intercept_ = np.polyfit(
                crowd[usable], ratings[usable].astype(float), 1
            )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        crowd = pd.to_numeric(features["crowd_shrunk"], errors="coerce").astype(float)
        predictions = self.slope_ * crowd + self.intercept_
        return np.where(crowd.notna() & (crowd > 0), predictions, self.intercept_)


class ContentRidge:
    """Ridge regression over TMDb content features.

    Ridge rather than anything larger because there are a few hundred training
    labels: the regularisation strength is doing most of the work, and it is
    chosen per fold by internal cross-validation rather than fixed.

    With ``debias=True`` the model is trained on popularity as usual but queried
    with popularity held at the training median, which cancels the portion of
    each prediction that came from fame. See ``features.neutralise_popularity``.
    """

    name = "content_ridge"

    def __init__(
        self,
        *,
        debias: bool = False,
        keyword_components: int = KEYWORD_COMPONENTS,
        synopsis_components: int = SYNOPSIS_COMPONENTS,
        blocks: tuple[str, ...] = FULL_BLOCKS,
    ) -> None:
        self.debias = debias
        self.keyword_components = keyword_components
        self.synopsis_components = synopsis_components
        self.blocks = blocks
        self.name = "content_ridge_debiased" if debias else "content_ridge"
        self.pipeline_: Pipeline | None = None
        self.train_features_: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> ContentRidge:
        self.pipeline_ = Pipeline(
            [
                (
                    "features",
                    build_preprocessor(
                        keyword_components=self.keyword_components,
                        synopsis_components=self.synopsis_components,
                        blocks=self.blocks,
                    ),
                ),
                ("ridge", RidgeCV(alphas=np.logspace(-1, 3.5, 24))),
            ]
        )
        self.pipeline_.fit(features, ratings.astype(float))
        self.train_features_ = features
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None or self.train_features_ is None:
            raise RuntimeError("fit before predict")
        if self.debias:
            features = neutralise_popularity(features, reference=self.train_features_)
        return self.pipeline_.predict(features)


class GradientBoosted:
    """Gradient-boosted trees over the same content features.

    A different inductive bias from ridge: trees capture interactions and
    thresholds a linear model cannot ("long *and* Japanese *and* pre-1970").

    Tempered expectations are warranted. Boosting wants far more than a few
    hundred rows, and the feature matrix here is mostly dense SVD components,
    which trees split awkwardly. It is included because the measurement should
    decide, not because it is expected to win.
    """

    name = "gradient_boosted"

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.6,
        min_child_weight: int = 5,
        reg_lambda: float = 2.0,
        keyword_components: int = KEYWORD_COMPONENTS,
        synopsis_components: int = SYNOPSIS_COMPONENTS,
        blocks: tuple[str, ...] = FULL_BLOCKS,
    ) -> None:
        self.blocks = blocks
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
        }
        self.keyword_components = keyword_components
        self.synopsis_components = synopsis_components
        self.pipeline_: Pipeline | None = None

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> GradientBoosted:
        from xgboost import XGBRegressor

        self.pipeline_ = Pipeline(
            [
                (
                    "features",
                    build_preprocessor(
                        keyword_components=self.keyword_components,
                        synopsis_components=self.synopsis_components,
                        blocks=self.blocks,
                    ),
                ),
                (
                    "xgb",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=0,
                        n_jobs=-1,
                        verbosity=0,
                        **self.params,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(features, ratings.astype(float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("fit before predict")
        return self.pipeline_.predict(features)


class ContentKNN:
    """Average the ratings of the nearest films in content space.

    The most literal statement of "you liked these, so you may like something
    like them", and a genuinely different bias from both ridge and trees: it is
    local, so a small cluster of films can carry a prediction without that
    pattern having to hold globally.

    Cosine distance, because the feature vectors are TF-IDF and SVD components
    where direction carries the meaning and magnitude mostly reflects how much
    metadata a film happens to have -- which is itself a popularity artefact.
    """

    name = "content_knn"

    def __init__(
        self,
        *,
        n_neighbors: int = 25,
        keyword_components: int = KEYWORD_COMPONENTS,
        synopsis_components: int = SYNOPSIS_COMPONENTS,
        blocks: tuple[str, ...] = FULL_BLOCKS,
    ) -> None:
        self.blocks = blocks
        self.n_neighbors = n_neighbors
        self.keyword_components = keyword_components
        self.synopsis_components = synopsis_components
        self.pipeline_: Pipeline | None = None

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> ContentKNN:
        from sklearn.neighbors import KNeighborsRegressor

        self.pipeline_ = Pipeline(
            [
                (
                    "features",
                    build_preprocessor(
                        keyword_components=self.keyword_components,
                        synopsis_components=self.synopsis_components,
                        blocks=self.blocks,
                    ),
                ),
                (
                    "knn",
                    KNeighborsRegressor(
                        n_neighbors=min(self.n_neighbors, max(1, len(features) - 1)),
                        metric="cosine",
                        weights="distance",
                    ),
                ),
            ]
        )
        self.pipeline_.fit(features, ratings.astype(float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("fit before predict")
        return self.pipeline_.predict(features)


class CollaborativeFold:
    """Fold the user into a latent space learned from other people's ratings.

    The classic cold-start-user solve: item factors come from MovieLens and are
    held fixed, and only the user's own position is fitted. Because the factors
    never saw this user's labels, the per-fold work is a small ridge on a
    64-dimensional input -- well determined by a few hundred ratings, where
    learning a representation from them would not be.

    Films the factor table does not cover fall back to the training mean. That is
    a real limitation rather than a bug: MovieLens has no usable signal for the
    least popular decile, so this model is deliberately silent there and the
    content model has to carry it.
    """

    name = "collaborative_fold"

    def __init__(self, factors: pd.DataFrame, links: pd.DataFrame) -> None:
        factor_columns = [c for c in factors.columns if c.startswith("f")]
        # The item mean is carried alongside the latent directions. Factorisation
        # is done on item-centred ratings so that popularity does not dominate
        # the components, but centring also removes how *well* a film is rated --
        # the strongest collaborative signal there is, and the one the crowd
        # baseline lives on. Dropping it leaves the model reconstructing quality
        # from taste directions alone, which it cannot do.
        self.columns = ["item_mean", *factor_columns]

        tmdb = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
        bridge = pd.DataFrame({"movieId": links["movieId"], "tmdb_id": tmdb}).dropna()
        table = bridge.merge(factors, on="movieId").drop_duplicates("tmdb_id")
        self.by_tmdb = table.set_index(table["tmdb_id"].astype("int64"))
        self.mean_ = 3.0
        self.ridge_: RidgeCV | None = None

    def _matrix(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ids = pd.to_numeric(features["tmdb_id"], errors="coerce").astype("Int64")
        known = ids.isin(self.by_tmdb.index).to_numpy()
        matrix = np.zeros((len(features), len(self.columns)), dtype="float32")
        if known.any():
            rows = self.by_tmdb.loc[ids[known].astype("int64"), self.columns]
            matrix[known] = rows.to_numpy(dtype="float32")
        return matrix, known

    def coverage(self, features: pd.DataFrame) -> np.ndarray:
        """Which films this model can actually speak about."""
        return self._matrix(features)[1]

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> CollaborativeFold:
        matrix, known = self._matrix(features)
        target = ratings.astype(float).to_numpy()
        self.mean_ = float(target.mean())
        if known.sum() >= 10:
            self.ridge_ = RidgeCV(alphas=np.logspace(-1, 3.5, 24))
            self.ridge_.fit(matrix[known], target[known])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix, known = self._matrix(features)
        predictions = np.full(len(features), self.mean_)
        if self.ridge_ is not None and known.any():
            predictions[known] = self.ridge_.predict(matrix[known])
        return predictions


class SupportGatedBlend:
    """Blend content and collaborative predictions, weighted by measured support.

    The two legs fail in different places, and the measurement said where: the
    collaborative leg is strong wherever MovieLens has depth and silent below
    roughly fifty ratings, while the content leg works everywhere but less
    sharply. So the blend weight is not a global constant -- it follows each
    film's actual collaborative coverage.

    Weights are fitted per fold by least squares over the two legs' out-of-fold
    behaviour on covered films, rather than being guessed.
    """

    name = "hybrid"

    def __init__(
        self, content: Model, collaborative: CollaborativeFold, *, inner_folds: int = 4
    ) -> None:
        self.content = content
        self.collaborative = collaborative
        self.inner_folds = inner_folds
        # Pristine copies, so each inner fold trains from scratch.
        self._content_template = copy.deepcopy(content)
        self._collaborative_template = copy.deepcopy(collaborative)
        self.weight_ = 0.5

    def _out_of_fold(
        self, features: pd.DataFrame, ratings: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Honest predictions from each leg, from models that never saw the row.

        Fitting blend weights on in-sample predictions does not work: the ridge
        fits its own training data far more closely than the collaborative leg
        fits its, so in-sample comparison hands the ridge all the weight and the
        blend collapses to one model. Weights have to be fitted on predictions
        made for unseen rows.
        """
        inner = KFold(n_splits=self.inner_folds, shuffle=True, random_state=0)
        content_oof = np.full(len(features), np.nan)
        collab_oof = np.full(len(features), np.nan)

        for train_idx, test_idx in inner.split(features):
            train_x, train_y = features.iloc[train_idx], ratings.iloc[train_idx]
            test_x = features.iloc[test_idx]
            content = copy.deepcopy(self._content_template).fit(train_x, train_y)
            collaborative = copy.deepcopy(self._collaborative_template).fit(train_x, train_y)
            content_oof[test_idx] = content.predict(test_x)
            collab_oof[test_idx] = collaborative.predict(test_x)
        return content_oof, collab_oof

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> SupportGatedBlend:
        target = ratings.astype(float).to_numpy()
        covered = self.collaborative.coverage(features)

        if covered.sum() >= 20 and len(features) >= self.inner_folds * 4:
            content_oof, collab_oof = self._out_of_fold(features, ratings)
            usable = covered & np.isfinite(content_oof) & np.isfinite(collab_oof)
            if usable.sum() >= 20:
                design = np.column_stack([content_oof[usable], collab_oof[usable]])
                weights, *_ = np.linalg.lstsq(design, target[usable], rcond=None)
                total = weights.sum()
                self.weight_ = float(np.clip(weights[1] / total, 0.0, 1.0)) if total else 0.5

        # Only now fit the legs on the full training set, for prediction.
        self.content.fit(features, ratings)
        self.collaborative.fit(features, ratings)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        content_p = self.content.predict(features)
        collab_p = self.collaborative.predict(features)
        covered = self.collaborative.coverage(features)
        # Uncovered films fall back to content alone rather than to a blend with
        # a constant, which would drag them toward the mean for no reason.
        blended = (1 - self.weight_) * content_p + self.weight_ * collab_p
        return np.where(covered, blended, content_p)


class StackedEnsemble:
    """Combine any number of models with weights fitted out of fold.

    The general form of the two-leg blend. Each base model is cross-validated
    *inside* the training data to get predictions for rows it did not see, and
    the combining weights are fitted on those. Weights are constrained to be
    non-negative, which keeps the result interpretable -- a weight is the share
    of the answer a model contributes -- and stops the fit from cancelling two
    correlated models against each other with large opposing coefficients, which
    is how stacking usually overfits on small data.

    Fitting on in-sample predictions instead would hand everything to whichever
    model overfits hardest; see the regression test in ``tests/test_hybrid.py``.
    """

    name = "ensemble"

    def __init__(
        self, models: list[Model], *, inner_folds: int = 4, name: str | None = None
    ) -> None:
        self.models = models
        self.inner_folds = inner_folds
        self.name = name or "ensemble(" + "+".join(m.name for m in models) + ")"
        self._templates = [copy.deepcopy(m) for m in models]
        self.weights_ = np.full(len(models), 1.0 / max(len(models), 1))
        self.intercept_ = 0.0

    def fit(self, features: pd.DataFrame, ratings: pd.Series) -> StackedEnsemble:
        from sklearn.linear_model import LinearRegression

        target = ratings.astype(float).to_numpy()
        if len(features) >= self.inner_folds * 4:
            inner = KFold(n_splits=self.inner_folds, shuffle=True, random_state=0)
            oof = np.full((len(features), len(self.models)), np.nan)
            for train_idx, test_idx in inner.split(features):
                train_x, train_y = features.iloc[train_idx], ratings.iloc[train_idx]
                test_x = features.iloc[test_idx]
                for position, template in enumerate(self._templates):
                    fitted = copy.deepcopy(template).fit(train_x, train_y)
                    oof[test_idx, position] = fitted.predict(test_x)

            usable = np.isfinite(oof).all(axis=1)
            if usable.sum() >= 20:
                combiner = LinearRegression(positive=True).fit(oof[usable], target[usable])
                self.weights_ = combiner.coef_
                self.intercept_ = float(combiner.intercept_)

        for model in self.models:
            model.fit(features, ratings)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        stacked = np.column_stack([m.predict(features) for m in self.models])
        return stacked @ self.weights_ + self.intercept_

    def describe_weights(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": [m.name for m in self.models],
                "weight": np.round(self.weights_, 3),
            }
        ).sort_values("weight", ascending=False, ignore_index=True)


#: Models that need no external data beyond TMDb metadata.
CONTENT_ONLY = (
    "global_mean",
    "crowd_score",
    "content_ridge",
    "content_ridge_debiased",
    "content_knn",
    "gradient_boosted",
)

#: Models that additionally need `lbrec factors`.
NEEDS_FACTORS = ("collaborative_fold", "hybrid", "ensemble", "ensemble_all")


def build_model(
    name: str, factors: pd.DataFrame | None = None, links: pd.DataFrame | None = None
) -> Model:
    """Construct one model by name. Raises for names needing absent data."""
    if name in NEEDS_FACTORS and (factors is None or links is None):
        raise ValueError(f"{name!r} needs MovieLens item factors -- run `lbrec factors` first")

    def collab() -> CollaborativeFold:
        return CollaborativeFold(factors, links)  # type: ignore[arg-type]

    builders: dict[str, callable] = {
        "global_mean": GlobalMean,
        "crowd_score": CrowdScore,
        "content_ridge": ContentRidge,
        "content_ridge_debiased": lambda: ContentRidge(debias=True),
        "content_knn": ContentKNN,
        "gradient_boosted": GradientBoosted,
        "collaborative_fold": collab,
        "hybrid": lambda: SupportGatedBlend(ContentRidge(), collab()),
        # Every learner that carries real signal, stacked out of fold.
        "ensemble": lambda: StackedEnsemble(
            [ContentRidge(), collab(), CrowdScore()], name="ensemble"
        ),
        "ensemble_all": lambda: StackedEnsemble(
            [ContentRidge(), collab(), CrowdScore(), ContentKNN(), GradientBoosted()],
            name="ensemble_all",
        ),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(sorted(builders))}")
    return builders[name]()


def available_models(*, with_factors: bool) -> list[str]:
    """Model names runnable given what data is present."""
    return list(CONTENT_ONLY) + (list(NEEDS_FACTORS) if with_factors else [])


def default_models(
    factors: pd.DataFrame | None = None, links: pd.DataFrame | None = None
) -> list[Model]:
    """The models compared by ``lbrec evaluate`` when none are named."""
    names = ["global_mean", "crowd_score", "content_ridge", "content_ridge_debiased"]
    if factors is not None and links is not None:
        names += ["collaborative_fold", "hybrid"]
    return [build_model(name, factors, links) for name in names]
