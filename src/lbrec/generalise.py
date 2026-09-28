"""Evaluate whether a model works for someone it has never seen.

Everything measured before this answered "how well does a model *fitted on
Volodya* predict Volodya". That is not the question. The question is whether a
model works for his sister with 40 ratings, or a stranger in Tashkent -- and a
per-user regression cannot answer it, because there is no shared model to test.

So the interface here is different from ``models.Model``, and the difference is
the whole point::

    fit_global(ratings_of_many_users)      # once, offline, shared by everyone
    score_user(given_items, given_ratings, target_items)   # history as *input*

Nothing is refitted per person. A new user's ratings are an argument, not a
training set, so scoring them is instant and works at any history length.

The protocol
------------
Users are split into train and test. **Item statistics and factors are learned
from train users only** -- a test user's ratings must never reach the global
fit, or the evaluation measures memorisation. This is the same leakage guard as
the per-fold one in ``evaluate``, one level up, and it is easy to get wrong
because the item side looks impersonal.

For each held-out user, ``n_given`` of their ratings are shown to the model and
the rest are predicted. Sweeping ``n_given`` gives the learning curve that
answers the practical question: how many ratings does a new person need before
this is worth anything?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

#: History sizes to sweep. Chosen around the numbers a real new user has: a
#: handful after signing up, a few dozen after an evening, a few hundred if they
#: have been logging for years.
GIVEN_SIZES = (10, 25, 50, 100, 200)

#: A test user needs enough ratings to both seed the model and score it.
MIN_TEST_RATINGS = 40


@dataclass(frozen=True)
class UserSplit:
    train_users: np.ndarray
    test_users: np.ndarray


def split_users(
    ratings: pd.DataFrame,
    *,
    n_test_users: int = 2000,
    min_ratings: int = MIN_TEST_RATINGS,
    seed: int = 0,
) -> UserSplit:
    """Hold out whole users, never individual ratings.

    Splitting ratings rather than users would leave every test user represented
    in the global fit, which is exactly the memorisation this protocol exists to
    rule out.
    """
    counts = ratings["userId"].value_counts()
    eligible = counts[counts >= min_ratings].index.to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_test_users, len(eligible)), replace=False)
    test = set(chosen.tolist())
    train = np.array([u for u in counts.index.to_numpy() if u not in test])
    return UserSplit(train_users=train, test_users=np.array(sorted(test)))


def profile_split(
    user_ratings: pd.DataFrame, n_given: int, *, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one user's ratings into what the model sees and what it predicts.

    Random rather than chronological: MovieLens timestamps are real, but a
    Letterboxd export's are logging dates, so a protocol that depended on
    ordering would not transfer to the users this is ultimately for.
    """
    shuffled = user_ratings.sample(frac=1.0, random_state=seed)
    return shuffled.iloc[:n_given], shuffled.iloc[n_given:]


class UserModel(Protocol):
    """A model trained once, applied to anyone."""

    name: str

    def fit_global(self, ratings: pd.DataFrame) -> UserModel:
        """Learn shared parameters from many users' ratings."""
        ...

    def score_user(
        self, given_items: np.ndarray, given_ratings: np.ndarray, target_items: np.ndarray
    ) -> np.ndarray:
        """Score candidate items for a user described only by their history."""
        ...


# --------------------------------------------------------------------------
# baselines and the fold-in model
# --------------------------------------------------------------------------
class UserMean:
    """Predict the user's own average. The floor, computed from their history."""

    name = "user_mean"

    def __init__(self) -> None:
        self.global_mean_ = 3.5

    def fit_global(self, ratings: pd.DataFrame) -> UserMean:
        self.global_mean_ = float(ratings["rating"].mean())
        return self

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        value = float(np.mean(given_ratings)) if len(given_ratings) else self.global_mean_
        return np.full(len(target_items), value)


class ItemMean:
    """Predict each item's average rating across the training users.

    The crowd baseline, honestly constructed: shrunk toward the global mean by
    how many ratings back it, so an item rated three times does not outrank one
    rated ten thousand times on the strength of a lucky sample.
    """

    name = "item_mean"

    def __init__(self, *, prior_weight: float = 20.0) -> None:
        self.prior_weight = prior_weight
        self.global_mean_ = 3.5
        self.item_mean_: pd.Series = pd.Series(dtype=float)

    def fit_global(self, ratings: pd.DataFrame) -> ItemMean:
        self.global_mean_ = float(ratings["rating"].mean())
        grouped = ratings.groupby("movieId")["rating"].agg(["sum", "count"])
        shrunk = (grouped["sum"] + self.prior_weight * self.global_mean_) / (
            grouped["count"] + self.prior_weight
        )
        self.item_mean_ = shrunk.astype("float32")
        return self

    def _items(self, target_items: np.ndarray) -> np.ndarray:
        return (
            pd.Series(target_items)
            .map(self.item_mean_)
            .fillna(self.global_mean_)
            .to_numpy(dtype=float)
        )

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        return self._items(target_items)


class BiasModel(ItemMean):
    """Item quality plus the user's own generosity: ``mu + b_i + b_u``.

    The smallest model that is genuinely personalised. The user term is one
    number read off their history at inference, which is why it costs nothing
    and works from the first rating.
    """

    name = "bias"

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        item_scores = self._items(target_items)
        if not len(given_ratings):
            return item_scores
        expected = self._items(given_items)
        user_offset = float(np.mean(np.asarray(given_ratings, dtype=float) - expected))
        return item_scores + user_offset


class BiasedMF:
    """The Netflix Prize model: ``mu + b_i + b_u + q_i . p_u``.

    Each term earns its place, and dropping one is not a simplification:

    ``mu``
        The global mean.
    ``b_i``
        Item bias -- how much better or worse than average a film is rated.
    ``b_u``
        User bias -- how generous a rater is. **An earlier version of this class
        omitted it**, and so lost to a model made of nothing else: a harsh rater
        received item-quality predictions with no downward shift, and the latent
        factors could not make that up.
    ``q_i . p_u``
        The interaction, and the only term that can express "likes Tarkovsky,
        dislikes Marvel". Without it every user gets the same ranking of films,
        merely shifted up or down.

    ``mu``, ``b_i`` and ``q_i`` are global and learned once; ``b_u`` and ``p_u``
    are solved at inference from a user's history. That split is the whole
    architecture: a stranger needs no training, only arithmetic.

    Biases are removed *before* factorising. Otherwise the leading components
    spend themselves re-encoding generosity and item quality, and the factors
    stop describing taste.

    Honest limitation: ``q_i`` comes from a truncated SVD, which treats an
    unrated film as rated exactly average. Funk's original trains on observed
    entries only and is better founded; that upgrade would change this class's
    internals but not its interface.
    """

    name = "biased_mf"

    def __init__(
        self,
        *,
        n_factors: int = 64,
        min_item_ratings: int = 20,
        item_prior: float = 20.0,
        user_prior: float = 10.0,
        reg: float = 8.0,
    ) -> None:
        self.n_factors = n_factors
        self.min_item_ratings = min_item_ratings
        self.item_prior = item_prior
        self.user_prior = user_prior
        self.reg = reg
        self.name = f"biased_mf_{n_factors}"
        self.global_mean_ = 3.5

    def fit_global(self, ratings: pd.DataFrame) -> BiasedMF:
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD

        self.global_mean_ = float(ratings["rating"].mean())

        # Item bias, shrunk so a film with three ratings cannot claim a large one.
        grouped = ratings.groupby("movieId")["rating"].agg(["sum", "count"])
        self.item_bias_ = (
            (grouped["sum"] - grouped["count"] * self.global_mean_)
            / (grouped["count"] + self.item_prior)
        ).astype("float32")

        keep = grouped["count"][grouped["count"] >= self.min_item_ratings].index
        subset = ratings[ratings["movieId"].isin(keep)]

        item_codes, item_ids = pd.factorize(subset["movieId"], sort=True)
        user_codes, user_ids = pd.factorize(subset["userId"], sort=False)
        values = subset["rating"].astype("float32").to_numpy()

        # Strip mu and b_i, then each training user's own b_u, before factorising.
        item_bias = self.item_bias_.reindex(item_ids).to_numpy(dtype="float32")
        after_item = values - self.global_mean_ - item_bias[item_codes]

        user_sums = np.zeros(len(user_ids), dtype="float32")
        np.add.at(user_sums, user_codes, after_item)
        user_counts = np.bincount(user_codes, minlength=len(user_ids)).astype("float32")
        user_bias = user_sums / (user_counts + self.user_prior)

        matrix = csr_matrix(
            (after_item - user_bias[user_codes], (user_codes, item_codes)),
            shape=(len(user_ids), len(item_ids)),
            dtype="float32",
        )
        svd = TruncatedSVD(n_components=self.n_factors, random_state=0).fit(matrix)

        self.item_index_ = pd.Series(np.arange(len(item_ids)), index=item_ids)
        self.factors_ = svd.components_.T.astype("float32")
        return self

    def _baseline(self, items: np.ndarray) -> np.ndarray:
        """``mu + b_i``, with unknown films falling back to the global mean."""
        return self.global_mean_ + pd.Series(items).map(self.item_bias_).fillna(0.0).to_numpy(
            dtype=float
        )

    def _rows(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = pd.Series(items).map(self.item_index_)
        known = positions.notna().to_numpy()
        matrix = np.zeros((len(items), self.n_factors), dtype="float32")
        if known.any():
            matrix[known] = self.factors_[positions[known].astype(int).to_numpy()]
        return matrix, known

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        given_items = np.asarray(given_items)
        target_items = np.asarray(target_items)
        baseline = self._baseline(target_items)
        if not len(given_items):
            return baseline

        # b_u: this user's average departure from mu + b_i, shrunk toward zero so
        # a three-rating history cannot claim a large personal offset.
        after_item = np.asarray(given_ratings, dtype=float) - self._baseline(given_items)
        user_bias = after_item.sum() / (len(after_item) + self.user_prior)
        baseline = baseline + user_bias

        given_matrix, given_known = self._rows(given_items)
        if given_known.sum() < 3:
            return baseline

        design = given_matrix[given_known].astype(float)
        residual = after_item[given_known] - user_bias
        gram = design.T @ design + self.reg * np.eye(self.n_factors)
        user_vector = np.linalg.solve(gram, design.T @ residual)

        target_matrix, _ = self._rows(target_items)
        return baseline + target_matrix.astype(float) @ user_vector


class ContentTower:
    """Score films by what they *are*, with the taste metric learned across users.

    The collaborative models share one blind spot: they can only score a film
    somebody in MovieLens has already rated. That is 84k films against a
    1.15M-film catalogue, and the missing 1.07M are exactly the obscure ones
    this project exists to surface. No amount of factorisation fixes that --
    the information simply is not there.

    This model uses different information entirely: TMDb metadata. It therefore
    fails in different places from the factorisation, which is the property that
    makes an ensemble of the two worth more than either.

    Shape, and why it generalises::

        profile_u = sum over rated films of (rating - expected) * features_i
        score(u, i) = mu + b_i + b_u + (w * profile_u) . features_i

    ``w`` is a per-dimension weight learned **once, across many training
    users** -- it says which content dimensions carry taste at all. The profile
    is a weighted average computed from a person's history at inference, so a
    stranger is scored by arithmetic, not by fitting.

    Weighting by ``rating - expected`` rather than by rating matters: a user who
    rates everything 4 would otherwise get a profile that simply reproduces the
    average film, because liking and watching would be indistinguishable.
    """

    name = "content_tower"

    def __init__(
        self,
        item_features: pd.DataFrame,
        *,
        item_prior: float = 20.0,
        user_prior: float = 10.0,
        learn_weights: bool = True,
        max_fit_users: int = 4000,
        reg: float = 1.0,
        seed: int = 0,
    ) -> None:
        """``item_features`` is indexed by movieId, one row of numbers per film."""
        self.features_ = item_features.astype("float32")
        # Unit-norm rows: otherwise films with more metadata dominate every
        # profile purely by having larger vectors, which is a popularity
        # artefact rather than a taste signal.
        norms = np.linalg.norm(self.features_.to_numpy(), axis=1, keepdims=True)
        self.features_.iloc[:, :] = self.features_.to_numpy() / np.maximum(norms, 1e-8)
        self.item_prior = item_prior
        self.user_prior = user_prior
        self.learn_weights = learn_weights
        self.max_fit_users = max_fit_users
        self.reg = reg
        self.seed = seed
        self.global_mean_ = 3.5
        self.weights_ = np.ones(self.features_.shape[1], dtype="float32")

    def fit_global(self, ratings: pd.DataFrame) -> ContentTower:
        self.global_mean_ = float(ratings["rating"].mean())
        grouped = ratings.groupby("movieId")["rating"].agg(["sum", "count"])
        self.item_bias_ = (
            (grouped["sum"] - grouped["count"] * self.global_mean_)
            / (grouped["count"] + self.item_prior)
        ).astype("float32")

        if self.learn_weights:
            self._learn_weights(ratings)
        return self

    def _vectors(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = self.features_.index.get_indexer(pd.Index(items))
        known = positions >= 0
        matrix = np.zeros((len(items), self.features_.shape[1]), dtype="float32")
        if known.any():
            matrix[known] = self.features_.to_numpy()[positions[known]]
        return matrix, known

    def _baseline(self, items: np.ndarray) -> np.ndarray:
        return self.global_mean_ + pd.Series(items).map(self.item_bias_).fillna(0.0).to_numpy(
            dtype=float
        )

    def _profile(self, items: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
        residual = values - self._baseline(items)
        user_bias = residual.sum() / (len(residual) + self.user_prior)
        vectors, known = self._vectors(items)
        if not known.any():
            return np.zeros(self.features_.shape[1], dtype="float32"), user_bias
        weights = (residual - user_bias)[known]
        profile = weights @ vectors[known]
        return (profile / max(known.sum(), 1)).astype("float32"), user_bias

    def _learn_weights(self, ratings: pd.DataFrame) -> None:
        """Learn which content dimensions carry taste, pooled over many users.

        Each training user contributes rows of ``(profile ⊙ item_features) ->
        residual``, built from a split of their own history so the profile never
        contains the rating being predicted. Ridge over the pooled rows gives one
        global weight vector.
        """
        rng = np.random.default_rng(self.seed)
        users = ratings["userId"].unique()
        chosen = rng.choice(users, size=min(self.max_fit_users, len(users)), replace=False)
        subset = ratings[ratings["userId"].isin(set(chosen.tolist()))]

        designs, targets = [], []
        for _, user_ratings in subset.groupby("userId"):
            if len(user_ratings) < 20:
                continue
            shuffled = user_ratings.sample(frac=1.0, random_state=self.seed)
            half = len(shuffled) // 2
            given, held = shuffled.iloc[:half], shuffled.iloc[half : half + 20]
            profile, user_bias = self._profile(
                given["movieId"].to_numpy(), given["rating"].to_numpy(dtype=float)
            )
            if not profile.any():
                continue
            vectors, known = self._vectors(held["movieId"].to_numpy())
            if not known.any():
                continue
            residual = (
                held["rating"].to_numpy(dtype=float)
                - self._baseline(held["movieId"].to_numpy())
                - user_bias
            )
            designs.append(vectors[known] * profile)
            targets.append(residual[known])

        if not designs:
            return
        design = np.vstack(designs).astype("float32")
        target = np.concatenate(targets).astype("float32")
        gram = design.T @ design + self.reg * np.eye(design.shape[1], dtype="float32")
        self.weights_ = np.linalg.solve(gram, design.T @ target).astype("float32")

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        given_items = np.asarray(given_items)
        target_items = np.asarray(target_items)
        baseline = self._baseline(target_items)
        if not len(given_items):
            return baseline

        profile, user_bias = self._profile(given_items, np.asarray(given_ratings, dtype=float))
        vectors, _ = self._vectors(target_items)
        return baseline + user_bias + vectors @ (self.weights_ * profile)


#: Films transformed per batch when vectorising the catalogue. Chosen for memory,
#: not speed: the cost is a few seconds, the saving is gigabytes.
_TRANSFORM_CHUNK = 10_000


def movielens_item_features(
    catalogue: pd.DataFrame, links: pd.DataFrame, *, n_components: int = 128
) -> pd.DataFrame:
    """Dense content vectors for MovieLens items, keyed by movieId.

    Fitted on the catalogue's metadata only -- no ratings are involved, so this
    can be built once and reused for every user without leaking anything.
    """
    from sklearn.decomposition import TruncatedSVD

    from lbrec.features import build_film_features, build_preprocessor

    bridge = links.copy()
    bridge["tmdbId"] = pd.to_numeric(bridge["tmdbId"], errors="coerce").astype("Int64")
    bridge = bridge.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")

    films = catalogue.copy()
    films["tmdb_id"] = films["tmdb_id"].astype("Int64")
    joined = bridge.merge(films, left_on="tmdbId", right_on="tmdb_id").drop_duplicates("movieId")
    if joined.empty:
        return pd.DataFrame()

    frame = build_film_features(joined)
    preprocessor = build_preprocessor().fit(frame)

    # Transformed in chunks into one preallocated float32 array. The preprocessor
    # emits dense blocks, and a single `fit_transform` over 84k films holds the
    # result plus sklearn's float64 intermediates at once -- several gigabytes for
    # a matrix that only needs one and a half. Fitting stays over all rows, so the
    # vocabularies and scalers are unchanged by the chunking.
    matrix: np.ndarray | None = None
    for start in range(0, len(frame), _TRANSFORM_CHUNK):
        stop = min(start + _TRANSFORM_CHUNK, len(frame))
        block = np.asarray(preprocessor.transform(frame.iloc[start:stop]), dtype="float32")
        if matrix is None:
            matrix = np.empty((len(frame), block.shape[1]), dtype="float32")
        matrix[start:stop] = block
    if matrix is None:
        return pd.DataFrame()

    usable = min(n_components, matrix.shape[1] - 1)
    if usable >= 1:
        matrix = TruncatedSVD(n_components=usable, random_state=0).fit_transform(matrix)
    return pd.DataFrame(
        np.asarray(matrix, dtype="float32"),
        index=pd.Index(joined["movieId"].to_numpy(), name="movieId"),
    )


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------
def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def evaluate_generalisation(
    models: list[UserModel],
    ratings: pd.DataFrame,
    split: UserSplit,
    *,
    given_sizes: tuple[int, ...] = GIVEN_SIZES,
    seed: int = 0,
    progress=None,
) -> pd.DataFrame:
    """Score every model on every held-out user, at each history length.

    Models are fitted **once** on the training users, then reused unchanged for
    every test user and every history length -- which is only possible because
    nothing about a user lives in the model.
    """
    train_ratings = ratings[ratings["userId"].isin(set(split.train_users.tolist()))]
    fitted = [model.fit_global(train_ratings) for model in models]

    test_ratings = ratings[ratings["userId"].isin(set(split.test_users.tolist()))]
    by_user = dict(tuple(test_ratings.groupby("userId")))

    records = []
    for n_given in given_sizes:
        for user_id, user_ratings in by_user.items():
            if len(user_ratings) <= n_given + 5:
                continue
            given, held = profile_split(user_ratings, n_given, seed=seed)
            given_items = given["movieId"].to_numpy()
            given_values = given["rating"].to_numpy(dtype=float)
            target_items = held["movieId"].to_numpy()
            actual = held["rating"].to_numpy(dtype=float)

            for model in fitted:
                predicted = np.asarray(
                    model.score_user(given_items, given_values, target_items), dtype=float
                )
                records.append(
                    {
                        "model": model.name,
                        "n_given": n_given,
                        "userId": user_id,
                        **_metrics(actual, np.clip(predicted, 0.5, 5.0)),
                    }
                )
        if progress is not None:
            progress.update(1)

    return pd.DataFrame.from_records(records)


def summarise(per_user: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE across held-out users, per model and history length.

    Averaged per user rather than pooled over ratings, so a handful of very
    active users cannot dominate the result -- the question is how well the
    model serves a *person*, not a rating.
    """
    if per_user.empty:
        return pd.DataFrame(columns=["model", "n_given", "users", "rmse", "rmse_sd"])
    return (
        per_user.groupby(["model", "n_given"], observed=True)
        .agg(users=("userId", "nunique"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"))
        .reset_index()
        .sort_values(["n_given", "rmse"], ignore_index=True)
    )
