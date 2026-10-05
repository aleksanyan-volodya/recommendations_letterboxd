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

#: What to do about ``b_i`` for a film the crowd has never rated. This is the
#: project's central fairness decision expressed as three testable options.
#:
#: ``known``     keep the crowd bias where it exists, zero elsewhere. Simple,
#:               and quietly unfair: an unrated film is pinned to "average"
#:               while a famous one keeps the bonus its votes bought it.
#: ``none``      drop ``b_i`` entirely and rank on content alone. Maximally fair
#:               to the tail, at the cost of discarding real quality evidence.
#: ``predicted`` infer ``b_i`` from content for unrated films, learned on the
#:               rated ones. Keeps the evidence without requiring fame, but it
#:               is a model of a model and will pull everything toward the mean.
ITEM_BIAS_KNOWN = "known"
ITEM_BIAS_NONE = "none"
ITEM_BIAS_PREDICTED = "predicted"
ITEM_BIAS_STRATEGIES = (ITEM_BIAS_KNOWN, ITEM_BIAS_NONE, ITEM_BIAS_PREDICTED)

#: A fourth option, which needs data the other three do not: take ``b_i`` from
#: a *second* crowd that has rated the film, calibrated onto this crowd's scale.
#: Where ``predicted`` guesses a bias from metadata, this one is measured --
#: just by different people. Films neither crowd rated fall back to
#: ``predicted``.
ITEM_BIAS_BORROWED = "borrowed"


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

    def __init__(
        self,
        item_features: pd.DataFrame,
        *,
        item_bias: str = ITEM_BIAS_KNOWN,
        borrowed_bias: pd.Series | None = None,
        bias_weight: float = 1.0,
        item_prior: float = 20.0,
        user_prior: float = 10.0,
        learn_weights: bool = True,
        max_fit_users: int = 4000,
        reg: float = 1.0,
        bias_reg: float = 10.0,
        seed: int = 0,
    ) -> None:
        """``item_features`` is indexed by movieId, one row of numbers per film.

        ``item_bias`` chooses what to do about films the crowd has never rated;
        see ``ITEM_BIAS_STRATEGIES``. It is the project's central fairness knob,
        not a tuning detail.

        ``borrowed_bias`` is a second crowd's item biases, indexed like
        ``item_features`` and on the same rating scale: each film's shrunk
        departure from *that* crowd's mean. Required by ``item_bias="borrowed"``,
        and when given it also reads the history (see ``_history_baseline``).

        ``bias_weight`` scales the crowd term when ranking candidates, tracing the
        frontier between the two corners: 1.0 trusts the crowd fully, 0.0 is
        exactly ``item_bias="none"`` and ranks on taste alone. It exists because
        the corners turned out to be a real trade -- the accurate settings show
        crowd-less films at a fraction of the rate users' own ratings justify, and
        the fair setting ranks far worse -- so the useful question is where on the
        curve to stand, not which end to pick.
        """
        known_strategies = (*ITEM_BIAS_STRATEGIES, ITEM_BIAS_BORROWED)
        if item_bias not in known_strategies:
            raise ValueError(
                f"unknown item_bias {item_bias!r}; expected one of {sorted(known_strategies)}"
            )
        if item_bias == ITEM_BIAS_BORROWED and borrowed_bias is None:
            raise ValueError("item_bias='borrowed' needs a borrowed_bias table")
        self.borrowed_bias = None if borrowed_bias is None else borrowed_bias.astype("float64")
        self.borrow_map_: tuple[float, float] | None = None
        self.borrow_fit_: dict[str, float] = {}
        self.features_ = item_features.astype("float32")
        # Unit-norm rows: otherwise films with more metadata dominate every
        # profile purely by having larger vectors, which is a popularity
        # artefact rather than a taste signal.
        norms = np.linalg.norm(self.features_.to_numpy(), axis=1, keepdims=True)
        self.features_.iloc[:, :] = self.features_.to_numpy() / np.maximum(norms, 1e-8)
        self.item_bias = item_bias
        self.bias_weight = float(bias_weight)
        self.item_prior = item_prior
        self.user_prior = user_prior
        self.learn_weights = learn_weights
        self.max_fit_users = max_fit_users
        self.reg = reg
        self.bias_reg = bias_reg
        self.seed = seed
        self.global_mean_ = 3.5
        self.weights_ = np.ones(self.features_.shape[1], dtype="float32")
        self.bias_weights_: np.ndarray | None = None

    @property
    def name(self) -> str:  # type: ignore[override]
        if self.bias_weight == 1.0:
            return f"tower_{self.item_bias}"
        return f"tower_{self.item_bias}_b{self.bias_weight:g}"

    def fit_global(self, ratings: pd.DataFrame) -> ContentTower:
        self.global_mean_ = float(ratings["rating"].mean())
        grouped = ratings.groupby("movieId")["rating"].agg(["sum", "count"])
        self.item_bias_ = (
            (grouped["sum"] - grouped["count"] * self.global_mean_)
            / (grouped["count"] + self.item_prior)
        ).astype("float32")
        self.item_count_ = grouped["count"]

        if self.borrowed_bias is not None:
            self._calibrate_borrowed()
        if self.item_bias in (ITEM_BIAS_PREDICTED, ITEM_BIAS_BORROWED):
            self._learn_item_bias()
        if self.learn_weights:
            self._learn_weights(ratings)
        return self

    def _calibrate_borrowed(self) -> None:
        """Map the second crowd's biases onto this crowd's scale.

        Two crowds disagree on scale, not only on films. On films both rated,
        MovieLens ``b_i`` is about 0.6 x Letterboxd's for well-known titles:
        cinephiles separate films more sharply. Uncalibrated, a borrowed bias
        would outrank a measured one on spread alone.

        The map is a line fitted on films both crowds rated, using only films
        with at least ``item_prior`` ratings here -- below that the target is
        mostly shrinkage, and the slope would learn the prior rather than the
        film. Only the training fit enters it, so no test user does.
        """
        assert self.borrowed_bias is not None
        supported = self.item_count_.index[self.item_count_ >= self.item_prior]
        shared = supported.intersection(self.borrowed_bias.index)
        if len(shared) < 3:
            return
        lent = self.borrowed_bias.loc[shared].to_numpy()
        own = self.item_bias_.loc[shared].to_numpy(dtype="float64")
        slope, intercept = np.polyfit(lent, own, 1)
        self.borrow_map_ = (float(intercept), float(slope))
        self.borrow_fit_ = {
            "films": float(len(shared)),
            "corr": float(np.corrcoef(lent, own)[0, 1]),
            "slope": float(slope),
            "intercept": float(intercept),
        }

    def _learn_item_bias(self) -> None:
        """Learn to predict a film's crowd bias from its content.

        Fitted only on films the crowd *has* rated, then applied to those it has
        not. Deliberately regularised hard: the honest output for an unknown film
        is a small, cautious offset, not a confident verdict. Films with little
        support are down-weighted so the fit is dominated by biases we trust.
        """
        shared = self.features_.index.intersection(self.item_bias_.index)
        if len(shared) < self.features_.shape[1] + 1:
            return
        design = self.features_.loc[shared].to_numpy().astype("float64")
        target = self.item_bias_.loc[shared].to_numpy().astype("float64")
        gram = design.T @ design + self.bias_reg * np.eye(design.shape[1])
        self.bias_weights_ = np.linalg.solve(gram, design.T @ target).astype("float32")

    def _item_bias_for(self, items: np.ndarray, *, strategy: str) -> np.ndarray:
        """The ``b_i`` term, and the whole popularity-fairness question.

        ``known`` leaves an unrated film at zero -- "exactly average" -- while a
        famous film keeps a large positive bias it earned from votes the obscure
        film never had the chance to receive. That is popularity bias entering
        through the back door, so the other strategies exist to escape it:
        ``none`` refuses the term entirely and ranks on content alone,
        ``predicted`` gives an unrated film the bias its metadata implies, and
        ``borrowed`` gives it the bias a second crowd measured.
        """
        if strategy == ITEM_BIAS_NONE:
            return np.zeros(len(items), dtype=float)

        bias = pd.Series(items).map(self.item_bias_).to_numpy(dtype=float)
        missing = np.isnan(bias)
        if strategy == ITEM_BIAS_BORROWED and missing.any() and self.borrow_map_ is not None:
            assert self.borrowed_bias is not None
            intercept, slope = self.borrow_map_
            lent = pd.Series(np.asarray(items)[missing]).map(self.borrowed_bias)
            bias[missing] = intercept + slope * lent.to_numpy(dtype=float)
            missing = np.isnan(bias)
        if strategy in (ITEM_BIAS_PREDICTED, ITEM_BIAS_BORROWED) and missing.any():
            if self.bias_weights_ is None:
                bias[missing] = 0.0
            else:
                vectors, _ = self._vectors(np.asarray(items)[missing])
                bias[missing] = vectors @ self.bias_weights_
        return np.nan_to_num(bias, nan=0.0)

    def _vectors(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = self.features_.index.get_indexer(pd.Index(items))
        known = positions >= 0
        matrix = np.zeros((len(items), self.features_.shape[1]), dtype="float32")
        if known.any():
            matrix[known] = self.features_.to_numpy()[positions[known]]
        return matrix, known

    def _baseline(self, items: np.ndarray) -> np.ndarray:
        """Expected rating for a *candidate*, under the chosen strategy.

        Damping by ``bias_weight`` deliberately decalibrates the prediction: the
        returned number is no longer an estimate of the rating, it is a ranking
        score that discounts reputation. RMSE therefore gets worse as the weight
        falls even where the ordering improves, which is why the ranking metrics
        are the ones to read on this curve.
        """
        bias = self._item_bias_for(np.asarray(items), strategy=self.item_bias)
        return self.global_mean_ + self.bias_weight * bias

    def _history_baseline(self, items: np.ndarray) -> np.ndarray:
        """Expected rating for a film the user has *already rated*.

        Always uses the crowd bias where it exists, whatever the strategy. Using
        the crowd to understand a person is not the same act as using it to rank
        what they see next, and only the second is the fairness problem: it is
        what lets a famous film outrank an obscure one on reputation alone.

        Withholding it here would instead corrupt the profile. The profile is
        built from ``rating - expected``, so without ``b_i`` a user who rates a
        masterpiece 5 looks enthusiastic rather than ordinary, and their taste
        vector drifts toward whatever is simply good.

        For the same reason it takes the best estimate on offer: a second
        crowd's measurement when one was given, a content guess otherwise.
        """
        strategy = ITEM_BIAS_PREDICTED if self.borrowed_bias is None else ITEM_BIAS_BORROWED
        return self.global_mean_ + self._item_bias_for(np.asarray(items), strategy=strategy)

    def _profile(self, items: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
        residual = values - self._history_baseline(items)
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

    def knows_crowd_opinion(self, items: np.ndarray) -> np.ndarray:
        """Whether the crowd has rated each film -- the real-world cold test.

        Not a flag from the experiment: absence from the item-bias table *is* what
        "nobody has rated this" means, both in the protocol and for the 1.07M
        catalogue films MovieLens has never seen. The stratified ranker can
        therefore use it at inference time for a stranger, not just in evaluation.

        A film only the *second* crowd rated still counts as unknown here. Its
        borrowed bias is measured, but by other people on a calibrated scale;
        whether that makes it comparable is exactly what ``tower_borrowed``
        against ``stratified_tower_borrowed`` measures.
        """
        return np.isin(np.asarray(items), self.item_bias_.index.to_numpy())


class StratifiedRanker:
    """Rank crowd-less and well-known films separately, then interleave.

    The frontier measurement showed that discounting the crowd bias cannot work:
    fairness only arrives at a weight of exactly zero, which throws away the
    ranking signal entirely. The reason is that ``b_i`` is not *comparable* across
    the two groups -- measured for a famous film, guessed or absent for an obscure
    one -- so the defect is in the comparison, not in the number. Within either
    group it is perfectly good information.

    So this never compares the two. Each group is ordered by the full model, and
    the two orderings are merged proportionally, which fixes the composition of
    the list by construction rather than by tuning. A film competes only against
    films whose evidence is of the same kind, and the quota comes from how many
    candidates are crowd-less -- known at inference time, and empirically within
    5% of the share users' own ratings put in their top ten.

    ``score_user`` still returns the calibrated prediction, because a discounted
    or interleaved score is no longer an estimate of a rating. Prediction and
    ranking are different jobs and this class only claims the second.
    """

    def __init__(self, base: ContentTower) -> None:
        self.base = base

    @property
    def name(self) -> str:
        return f"stratified_{self.base.name}"

    def fit_global(self, ratings: pd.DataFrame) -> StratifiedRanker:
        self.base.fit_global(ratings)
        return self

    def score_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        return self.base.score_user(given_items, given_ratings, target_items)

    def rank_user(self, given_items, given_ratings, target_items) -> np.ndarray:
        scores = self.base.score_user(given_items, given_ratings, target_items)
        known = self.base.knows_crowd_opinion(np.asarray(target_items))
        return _interleave(scores, known)


def _interleave(scores: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Merge two orderings so every prefix holds each group at its own rate.

    Each film is placed by its *relative* standing inside its own group -- the
    best of 20 crowd-less films sits at 0.025, the best of 80 known films also at
    0.0125 -- so sorting by that position alternates between the groups in
    proportion to their sizes. A top ten then contains crowd-less films at the
    rate they appear among the candidates, whatever the score scales look like.

    Returned as a descending score rather than an order, so callers can treat it
    like any other ranking score.
    """
    position = np.empty(len(scores), dtype=float)
    for member in (group, ~group):
        size = int(member.sum())
        if not size:
            continue
        # Ties broken by a fixed shuffle: candidate arrays arrive movieId-ordered,
        # and movieId tracks release year, so ties would go to the oldest film.
        index = np.flatnonzero(member)
        shuffled = np.random.default_rng(0).permutation(len(index))
        order = index[shuffled[np.argsort(-scores[index][shuffled], kind="stable")]]
        position[order] = (np.arange(size) + 0.5) / size
    return -position


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


def cold_item_split(
    ratings: pd.DataFrame, *, share: float = 0.2, seed: int = 0, min_ratings: int = 5
) -> np.ndarray:
    """Choose films to hide from the crowd, simulating the unrated tail.

    The project's real question cannot be asked of MovieLens directly: every film
    in it has been rated, so there is no way to see how a model handles the 1.07M
    films it has never heard of. So we manufacture the situation -- pick films,
    delete their ratings from the global fit, and then ask held-out users about
    exactly those films.

    Films are chosen at random rather than from the tail on purpose. Tail films
    are *also* films with few ratings, so selecting them would confound "the
    crowd is silent" with "the ground truth is noisy", and a difference between
    strategies could not be attributed to either. Random selection isolates the
    silence, and the support of each film is reported alongside so the
    interaction stays visible.

    ``min_ratings`` keeps out films so thinly rated that their held-out ratings
    are mostly noise -- we are testing the model, not the ground truth.
    """
    counts = ratings["movieId"].value_counts()
    eligible = counts[counts >= min_ratings].index.to_numpy()
    rng = np.random.default_rng(seed)
    size = int(round(share * len(eligible)))
    return np.sort(rng.choice(eligible, size=max(size, 1), replace=False))


def evaluate_cold_items(
    models: list[UserModel],
    ratings: pd.DataFrame,
    split: UserSplit,
    cold_items: np.ndarray,
    *,
    given_sizes: tuple[int, ...] = GIVEN_SIZES,
    seed: int = 0,
    progress=None,
) -> pd.DataFrame:
    """Rank crowd-less films *against* well-known ones, which is the real task.

    Two deletions, and both are necessary. Cold films are removed from the
    training ratings, so no item bias or latent factor can encode them; and they
    are removed from each test user's *given* history, so a model cannot learn
    about a cold film from the very user it is being scored on.

    The candidate set deliberately mixes cold and warm films. Scoring cold films
    in isolation looks like the right experiment and is not: with ``b_i`` absent
    for every candidate, a missing-bias strategy has nothing to differ about, and
    ``known`` and ``none`` produce identical numbers. Popularity bias is not a
    property of how a film is scored, it is a property of *what it loses to* --
    so it only becomes visible when an unknown film has to outrank a famous one.

    Hence ``tail_share``: of the films this model puts in a user's top ten, how
    many are crowd-less? Compared against ``tail_share_actual`` -- how many the
    user themselves would put there -- it measures suppression of the tail
    directly, in a way no RMSE can.
    """
    cold = set(np.asarray(cold_items).tolist())
    train_ratings = ratings[ratings["userId"].isin(set(split.train_users.tolist()))]
    train_ratings = train_ratings[~train_ratings["movieId"].isin(cold)]
    fitted = [model.fit_global(train_ratings) for model in models]

    test_ratings = ratings[ratings["userId"].isin(set(split.test_users.tolist()))]
    by_user = dict(tuple(test_ratings.groupby("userId")))

    records = []
    for n_given in given_sizes:
        for user_id, user_ratings in by_user.items():
            is_cold = user_ratings["movieId"].isin(cold).to_numpy()
            warm_rows = user_ratings[~is_cold]
            if len(warm_rows) < n_given + 5 or not is_cold.any():
                continue

            # History is warm-only; targets are everything else, cold and warm
            # together, so the two compete for the same top of the ranking.
            given = warm_rows.sample(n=n_given, random_state=seed)
            held = user_ratings.drop(index=given.index)
            given_items = given["movieId"].to_numpy()
            given_values = given["rating"].to_numpy(dtype=float)
            target_items = held["movieId"].to_numpy()
            actual = held["rating"].to_numpy(dtype=float)
            target_cold = held["movieId"].isin(cold).to_numpy()
            if not target_cold.any():
                continue

            for model in fitted:
                predicted = np.clip(
                    np.asarray(
                        model.score_user(given_items, given_values, target_items), dtype=float
                    ),
                    0.5,
                    5.0,
                )
                # A model may rank by something other than its own prediction --
                # an interleaved order is not a rating estimate. RMSE stays on the
                # calibrated value, the ranking metrics use whatever it ranks by.
                ranker = getattr(model, "rank_user", None)
                ranked = (
                    predicted
                    if ranker is None
                    else np.asarray(ranker(given_items, given_values, target_items), dtype=float)
                )
                records.append(
                    {
                        "model": model.name,
                        "n_given": n_given,
                        "userId": user_id,
                        "cold_share": float(target_cold.mean()),
                        **_metrics(actual, predicted),
                        "rmse_cold": _rmse(actual[target_cold], predicted[target_cold]),
                        "rmse_warm": _rmse(actual[~target_cold], predicted[~target_cold]),
                        **_ranking_metrics(actual, ranked, target_cold),
                    }
                )
        if progress is not None:
            progress.update(1)

    return pd.DataFrame.from_records(records)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    if not len(actual):
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


#: How many films a recommendation list actually shows someone.
TOP_K = 10


def _ranking_metrics(
    actual: np.ndarray, predicted: np.ndarray, is_cold: np.ndarray, *, k: int = TOP_K
) -> dict[str, float]:
    """How well the ordering is recovered, and who wins the top of the list.

    RMSE is the wrong sole metric for a recommender: being wrong by 0.3 on every
    film matters far less than putting the wrong film first. Spearman needs at
    least three ratings and some spread to mean anything, so it is NaN rather
    than misleading when a user rated too few held-out films.

    ``tail_share`` is the fairness number. Against ``tail_share_actual`` it says
    whether crowd-less films reach the top of the list as often as the user's own
    ratings say they deserve to.
    """
    metrics: dict[str, float] = {"spearman": float("nan")}
    if len(actual) >= 3 and len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
        ranked_actual = pd.Series(actual).rank().to_numpy()
        ranked_predicted = pd.Series(predicted).rank().to_numpy()
        metrics["spearman"] = float(np.corrcoef(ranked_actual, ranked_predicted)[0, 1])

    top = min(k, len(actual))
    # Ties broken by a fixed shuffle, not by array order: MovieLens rows arrive
    # movieId-ordered, and movieId correlates with age, so `argsort` alone would
    # quietly hand every tie to the oldest film.
    order = np.random.default_rng(0).permutation(len(actual))
    by_predicted = order[np.argsort(-predicted[order], kind="stable")][:top]
    by_actual = order[np.argsort(-actual[order], kind="stable")][:top]
    metrics["tail_share"] = float(is_cold[by_predicted].mean())
    metrics["tail_share_actual"] = float(is_cold[by_actual].mean())
    return metrics


#: How far below the deserved tail share still counts as fair. 10% is tight
#: enough that a model cannot pass by mostly ignoring the tail, and loose enough
#: not to reward the noise in a single user's top ten.
FAIR_TOLERANCE = 0.1

#: Crowd-less share of the candidate set over which `StratifiedRanker` has been
#: measured fair: mean per-user ``cold_share`` in the cold-items run, 0.250 at
#: 10 given ratings to 0.461 at 200. Its quota is that share, so outside this
#: range the guarantee is extrapolated, not measured -- over the raw 1.15M
#: catalogue (93% crowd-less) it would fill a top ten with unrated films.
VALIDATED_COLD_SHARE = (0.25, 0.46)


def frontier(per_user: pd.DataFrame, *, tolerance: float = FAIR_TOLERANCE) -> pd.DataFrame:
    """Rank models by the project's actual objective, not by RMSE.

    The goal was stated as a constraint and an objective, not a preference on a
    curve: crowd-less films should reach the top of a list about as often as the
    user's own ratings say they deserve -- and *subject to that*, the ranking
    should be as good as possible.

    So this reports, per history length, whether each model clears the fairness
    bar and how well it ranks, with the best fair model marked. A model that
    ranks superbly while suppressing the tail does not win here, and neither does
    a perfectly fair model that cannot order anything: the first fails the
    constraint, the second loses on the objective.

    Overshooting is not penalised, but it is reported: showing crowd-less films
    *more* often than they deserve is promoting obscurity for its own sake, which
    is a different product from the one we are building.
    """
    required = ["model", "n_given", "spearman", "tail_share", "tail_share_actual"]
    missing = [column for column in required if column not in per_user]
    if missing:
        raise ValueError(f"per-user results lack {missing}; run the cold-item protocol")

    table = (
        per_user.groupby(["model", "n_given"], observed=True)
        .agg(
            users=("userId", "nunique"),
            spearman=("spearman", "mean"),
            tail_share=("tail_share", "mean"),
            deserved=("tail_share_actual", "mean"),
        )
        .reset_index()
    )
    table["fairness"] = table["tail_share"] / table["deserved"]
    table["fair"] = table["fairness"] >= 1.0 - tolerance
    table = table.sort_values(["n_given", "fair", "spearman"], ascending=[True, False, False])

    best = (
        table[table["fair"]]
        .groupby("n_given", observed=True)["spearman"]
        .idxmax()
        .pipe(lambda index: table.index.isin(index))
    )
    table["best"] = best
    return table.reset_index(drop=True)


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
    aggregations = {
        "users": ("userId", "nunique"),
        "rmse": ("rmse", "mean"),
        "rmse_sd": ("rmse", "std"),
    }
    for column in ("rmse_cold", "rmse_warm", "spearman", "tail_share", "tail_share_actual"):
        if column in per_user:
            aggregations[column] = (column, "mean")
    return (
        per_user.groupby(["model", "n_given"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["n_given", "rmse"], ignore_index=True)
    )
