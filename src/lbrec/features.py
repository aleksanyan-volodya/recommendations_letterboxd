"""Turn TMDb metadata into a feature matrix.

Every transformer here is fitted **inside** a cross-validation fold rather than
over the whole dataset. With 786 labels the temptation to fit a vectoriser or a
scaler once and reuse it is strong, and it leaks: a keyword vocabulary or a
column mean derived from the held-out films quietly inflates every score. The
pipeline exists so that cannot happen by accident.

Popularity (``log_votes``) and the crowd score (``vote_average``) are included
deliberately, even though the eventual goal is to *remove* their influence.
Having them as explicit columns is what makes debiasing possible: a model can be
trained with them and then queried with them held constant, which cancels the
part of a prediction that came from fame rather than from the film. Dropping
them instead would only hide the bias in correlated features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: Columns of list type that become bag-of-words documents.
LIST_COLUMNS = ("genres", "keywords", "directors", "cast", "production_countries")

NUMERIC_COLUMNS = ["year", "runtime", "log_votes", "crowd_shrunk"]

#: Prior for shrinking the crowd score, measured from films with >=100 votes.
CROWD_PRIOR_MEAN = 6.5

#: How many votes of evidence the prior is worth. A film needs roughly this many
#: votes before its own average outweighs the prior.
CROWD_PRIOR_WEIGHT = 50.0


def shrink_crowd_score(
    vote_average: pd.Series,
    vote_count: pd.Series,
    *,
    prior_mean: float = CROWD_PRIOR_MEAN,
    prior_weight: float = CROWD_PRIOR_WEIGHT,
) -> pd.Series:
    """Crowd score shrunk toward the global mean by how many votes back it.

    ``vote_average`` is unusable raw. In the real catalogue 3,093 films have a
    single vote of 10/10, and 87,151 have no votes at all -- where TMDb records
    the average as 0.0, which a model reads as "terrible" rather than "unknown".
    Ranking on the raw column therefore surfaces films that exactly one person
    has ever rated, which is what the first recommendation run actually did.

    Cross-validation could not have caught this: every film in a user's library
    has thousands of votes, so the pathology only exists in the catalogue. It is
    the clearest example of why offline metrics on self-selected films do not
    transfer to the population we recommend from.

    The standard weighted-rating formula::

        (v * R + m * C) / (v + m)

    With no votes it returns the prior exactly, and with many votes it returns
    the film's own average.
    """
    votes = pd.to_numeric(vote_count, errors="coerce").astype("Float64").astype(float).fillna(0.0)
    average = pd.to_numeric(vote_average, errors="coerce").astype("Float64").astype(float)
    # A recorded average of 0 alongside 0 votes means "unknown", not "zero".
    average = average.fillna(prior_mean)
    return (votes * average + prior_weight * prior_mean) / (votes + prior_weight)


CATEGORICAL_COLUMNS = ["original_language"]

#: Keyword space is large (8k+ distinct terms against a few hundred films), so
#: it is reduced rather than fed in raw.
KEYWORD_COMPONENTS = 96

#: Synopsis text is reduced harder than keywords: it is noisier per dimension.
SYNOPSIS_COMPONENTS = 64


def _join(values) -> str:
    """Flatten a list column into a space-joined document of single tokens.

    Multi-word names become one token ("andrei_tarkovsky") so that the
    vectoriser treats a person as an atom rather than as their first name plus
    their surname.
    """
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return ""
    return " ".join(str(v).strip().lower().replace(" ", "_") for v in values if str(v).strip())


def build_film_features(films: pd.DataFrame) -> pd.DataFrame:
    """One row per film, with list columns flattened into text documents.

    Uses only metadata, never ratings, so it can be computed once for the whole
    catalogue without leaking anything about a particular user's labels.
    """
    frame = pd.DataFrame(index=films.index)
    frame["tmdb_id"] = films["tmdb_id"].astype("Int64")
    for column in LIST_COLUMNS:
        frame[column] = films[column].map(_join) if column in films else ""

    frame["year"] = pd.to_numeric(films.get("year"), errors="coerce").astype("Float64")
    frame["runtime"] = pd.to_numeric(films.get("runtime"), errors="coerce").astype("Float64")
    votes = pd.to_numeric(films.get("vote_count"), errors="coerce").astype("Float64")
    frame["log_votes"] = np.log1p(votes.astype(float))
    frame["vote_average"] = pd.to_numeric(films.get("vote_average"), errors="coerce").astype(
        "Float64"
    )
    # The usable form of the crowd score. See shrink_crowd_score for why the raw
    # column cannot be ranked on.
    frame["crowd_shrunk"] = shrink_crowd_score(frame["vote_average"], votes)
    frame["original_language"] = films.get(
        "original_language", pd.Series(index=films.index)
    ).fillna("unknown")

    # The synopsis is the one rich feature that does not thin out with
    # obscurity: in the reference export the least popular decile has a median
    # of zero keywords but an overview of normal length. It is therefore the
    # only content signal available for the films this project most cares about.
    overview = films.get("overview", pd.Series("", index=films.index)).fillna("")
    tagline = films.get("tagline", pd.Series("", index=films.index)).fillna("")
    frame["synopsis"] = (overview.astype(str) + " " + tagline.astype(str)).str.strip().str.lower()
    return frame


class TextBlock(BaseEstimator, TransformerMixin):
    """TF-IDF, optionally reduced, that degrades instead of raising.

    Sparse metadata is the normal case here, not an edge case: a fold may
    contain no film with a synopsis, or none with a shared keyword, and
    ``TfidfVectorizer`` raises "empty vocabulary" rather than returning nothing.
    A crash in that situation would mean the pipeline fails precisely for the
    obscure films the project exists to serve, so an empty block contributes a
    zero column instead.

    The component count is likewise clamped to what the fitted vocabulary can
    support, since ``TruncatedSVD`` cannot ask for more dimensions than features.

    SVD output is standardised afterwards. Ridge applies a single penalty to
    every coefficient, so a block whose columns happen to be large is
    effectively regularised *less* than one whose columns are small. Raw SVD
    components carry the singular values and so dwarf L2-normalised TF-IDF and
    standardised numerics sitting beside them -- an arbitrary weighting, not a
    modelling decision.
    """

    def __init__(self, *, components: int | None = None, **vectorizer_kwargs) -> None:
        self.components = components
        self.vectorizer_kwargs = vectorizer_kwargs

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        self.vectorizer_ = TfidfVectorizer(**self.vectorizer_kwargs)
        self.svd_ = None
        self.scaler_ = None
        try:
            matrix = self.vectorizer_.fit_transform(X)
        except ValueError:  # empty vocabulary
            self.vectorizer_ = None
            return self
        if self.components:
            usable = min(self.components, matrix.shape[1] - 1)
            if usable >= 1:
                self.svd_ = TruncatedSVD(n_components=usable, random_state=0).fit(matrix)
                self.scaler_ = StandardScaler().fit(self.svd_.transform(matrix))
        return self

    def transform(self, X):  # noqa: N803
        if self.vectorizer_ is None:
            return np.zeros((len(X), 1))
        matrix = self.vectorizer_.transform(X)
        if self.svd_ is None:
            return matrix.toarray()
        reduced = self.svd_.transform(matrix)
        return self.scaler_.transform(reduced) if self.scaler_ is not None else reduced


def _text_pipeline(max_features: int, components: int | None = None) -> TextBlock:
    return TextBlock(
        components=components,
        max_features=max_features,
        min_df=2,  # a term seen once cannot generalise
        token_pattern=r"[^ ]+",  # tokens are pre-joined; do not re-split names
    )


#: Blocks available for every film in the export.
FULL_BLOCKS = (
    "genres",
    "keywords",
    "directors",
    "cast",
    "countries",
    "synopsis",
    "numeric",
    "language",
)

#: Blocks available for every film in the candidate catalogue. The crowd dump
#: carries no keywords, cast or crew, so a model meant to score the catalogue
#: must be *trained* without them too -- otherwise it applies coefficients
#: learned on features that are uniformly absent at inference time, which is a
#: train/inference mismatch rather than a recommendation.
CATALOGUE_BLOCKS = ("genres", "countries", "synopsis", "numeric", "language")


def build_preprocessor(
    *,
    keyword_components: int = KEYWORD_COMPONENTS,
    synopsis_components: int = SYNOPSIS_COMPONENTS,
    blocks: tuple[str, ...] = FULL_BLOCKS,
) -> ColumnTransformer:
    """The feature pipeline, unfitted.

    Returned unfitted on purpose: the caller fits it per fold, so vocabularies
    and scaler statistics never see held-out films.

    ``blocks`` selects which feature groups to build; see ``CATALOGUE_BLOCKS``.
    """
    unknown = set(blocks) - set(FULL_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature block(s): {sorted(unknown)}")

    definitions = {
        "genres": (_text_pipeline(max_features=64), "genres"),
        "keywords": (
            _text_pipeline(max_features=6000, components=keyword_components),
            "keywords",
        ),
        "directors": (_text_pipeline(max_features=1500), "directors"),
        "cast": (_text_pipeline(max_features=3000), "cast"),
        "countries": (_text_pipeline(max_features=120), "production_countries"),
        # Natural language, so unlike the pre-joined columns it needs real word
        # tokenisation, stop-word removal and dimensionality reduction.
        "synopsis": (
            TextBlock(
                components=synopsis_components,
                max_features=20000,
                min_df=3,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
            "synopsis",
        ),
        "numeric": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            NUMERIC_COLUMNS,
        ),
        "language": (
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5),
            CATEGORICAL_COLUMNS,
        ),
    }
    return ColumnTransformer(
        [(name, *definitions[name]) for name in blocks],
        remainder="drop",
        sparse_threshold=0.0,
    )


def neutralise_popularity(features: pd.DataFrame, *, reference: pd.DataFrame) -> pd.DataFrame:
    """Replace popularity with a constant, keeping every other feature intact.

    This is the inference half of the "train with it, predict without it" trick:
    the model learns how much of a rating is explained by fame, and then that
    term is switched off by feeding every film the same popularity. What remains
    is the part of the prediction attributable to the film itself.

    ``reference`` supplies the value to hold constant -- the training median, so
    the substituted value is one the model actually saw.
    """
    neutral = features.copy()
    neutral["log_votes"] = float(reference["log_votes"].median())
    return neutral
