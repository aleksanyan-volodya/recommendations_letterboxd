"""Tests for the user-generalisation protocol.

This protocol exists to answer "does the model work for someone it has never
seen", so the property that matters most is that a test user's ratings never
reach the global fit. That leakage is easy to introduce and invisible in the
output -- item statistics look impersonal right up until they encode the person
you are about to score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbrec.generalise import (
    BiasedMF,
    BiasModel,
    ItemMean,
    UserMean,
    evaluate_generalisation,
    profile_split,
    split_users,
    summarise,
)


def make_ratings(n_users: int = 60, n_items: int = 80, seed: int = 0) -> pd.DataFrame:
    """Synthetic ratings with real structure: item quality plus user taste."""
    rng = np.random.default_rng(seed)
    item_quality = rng.normal(3.5, 0.6, n_items)
    item_axis = rng.normal(0, 1, n_items)  # one latent taste dimension
    rows = []
    for user in range(n_users):
        taste = rng.normal(0, 1)
        generosity = rng.normal(0, 0.3)
        # Cap at the catalogue size so small fixtures stay valid.
        n_watched = min(int(rng.integers(50, 70)), n_items)
        items = rng.choice(n_items, size=n_watched, replace=False)
        for item in items:
            score = item_quality[item] + generosity + taste * item_axis[item]
            rows.append(
                {
                    "userId": user,
                    "movieId": int(item),
                    "rating": float(np.clip(score + rng.normal(0, 0.2), 0.5, 5.0)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def test_split_holds_out_whole_users_not_ratings():
    """Splitting ratings would leave every test user inside the global fit."""
    ratings = make_ratings()
    split = split_users(ratings, n_test_users=10, min_ratings=10, seed=0)
    assert set(split.train_users).isdisjoint(set(split.test_users))
    assert len(split.test_users) == 10


def test_split_only_holds_out_users_with_enough_ratings():
    """A test user needs enough ratings to both seed the model and score it."""
    ratings = make_ratings()
    thin = pd.DataFrame({"userId": [999] * 5, "movieId": range(5), "rating": [3.0] * 5})
    split = split_users(pd.concat([ratings, thin]), n_test_users=50, min_ratings=40, seed=0)
    assert 999 not in set(split.test_users)


def test_profile_split_gives_the_model_exactly_n_ratings():
    ratings = make_ratings()
    user = ratings[ratings["userId"] == 0]
    given, held = profile_split(user, 10, seed=0)
    assert len(given) == 10
    assert len(held) == len(user) - 10
    assert set(given.index).isdisjoint(set(held.index))


# --------------------------------------------------------------------------
# the leakage guard
# --------------------------------------------------------------------------
def test_global_fit_never_sees_a_test_user():
    """The guard this whole protocol rests on."""
    ratings = make_ratings()
    split = split_users(ratings, n_test_users=8, min_ratings=40, seed=0)
    test_users = set(split.test_users.tolist())
    seen: list[set] = []

    class Spy:
        name = "spy"

        def fit_global(self, frame):
            seen.append(set(frame["userId"].unique().tolist()))
            return self

        def score_user(self, given_items, given_ratings, target_items):
            return np.full(len(target_items), 3.5)

    evaluate_generalisation([Spy()], ratings, split, given_sizes=(10,))
    assert seen, "fit_global was never called"
    assert seen[0].isdisjoint(test_users), "a held-out user reached the global fit"


def test_models_are_fitted_once_not_per_user():
    """Refitting per user would be the architecture this protocol replaces."""
    ratings = make_ratings()
    split = split_users(ratings, n_test_users=6, min_ratings=40, seed=0)
    calls = {"fit": 0, "score": 0}

    class Counter:
        name = "counter"

        def fit_global(self, frame):
            calls["fit"] += 1
            return self

        def score_user(self, given_items, given_ratings, target_items):
            calls["score"] += 1
            return np.full(len(target_items), 3.5)

    evaluate_generalisation([Counter()], ratings, split, given_sizes=(10, 25))
    assert calls["fit"] == 1
    assert calls["score"] > 1


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------
def test_user_mean_uses_the_history_it_is_given():
    model = UserMean().fit_global(make_ratings())
    scores = model.score_user(np.array([1, 2]), np.array([4.0, 5.0]), np.array([7, 8, 9]))
    assert scores == pytest.approx(np.full(3, 4.5))


def test_user_mean_falls_back_when_there_is_no_history():
    model = UserMean().fit_global(make_ratings())
    scores = model.score_user(np.array([]), np.array([]), np.array([1, 2]))
    assert np.isfinite(scores).all()


def test_item_mean_is_shrunk_toward_the_global_mean():
    """An item rated three times must not outrank one rated ten thousand times."""
    ratings = pd.DataFrame(
        {
            "userId": list(range(3)) + list(range(200)),
            "movieId": [1] * 3 + [2] * 200,
            "rating": [5.0] * 3 + [4.0] * 200,
        }
    )
    model = ItemMean(prior_weight=20.0).fit_global(ratings)
    thin, thick = model.score_user(np.array([]), np.array([]), np.array([1, 2]))
    assert thin < 5.0  # pulled well back from its lucky sample
    assert thick == pytest.approx(4.0, abs=0.05)  # barely moved


def test_unknown_items_fall_back_rather_than_failing():
    """A brand-new film has no training ratings; it must still get a number."""
    model = ItemMean().fit_global(make_ratings())
    scores = model.score_user(np.array([1]), np.array([4.0]), np.array([10**9]))
    assert np.isfinite(scores).all()


def test_bias_model_personalises_from_the_history_alone():
    """A generous user's predictions must shift up, with no refitting."""
    ratings = make_ratings()
    model = BiasModel().fit_global(ratings)
    items = ratings["movieId"].unique()[:5]
    expected = model._items(items)

    generous = model.score_user(items, expected + 1.0, items)
    stingy = model.score_user(items, expected - 1.0, items)
    assert (generous > stingy).all()


def test_factorisation_learns_taste_not_just_item_quality():
    """Two users with opposite taste must get different rankings of the same films."""
    ratings = make_ratings(n_users=120, n_items=90, seed=1)
    split = split_users(ratings, n_test_users=2, min_ratings=40, seed=1)
    model = BiasedMF(n_factors=8, min_item_ratings=3).fit_global(
        ratings[ratings["userId"].isin(set(split.train_users.tolist()))]
    )

    items = ratings["movieId"].unique()[:40]
    # Build two histories that disagree sharply about the same films.
    history = np.concatenate([items[:10], items[10:20]])
    one = model.score_user(history, np.concatenate([np.full(10, 5.0), np.full(10, 1.0)]), items)
    other = model.score_user(history, np.concatenate([np.full(10, 1.0), np.full(10, 5.0)]), items)

    # Compare the *taste* component, not the total. Both scores sit on the same
    # item-mean baseline, which dominates their variance and would make even
    # opposite tastes correlate at ~0.99.
    neutral = model.score_user(np.array([]), np.array([]), items)
    assert not np.allclose(one, neutral), "the model ignored the history entirely"
    assert np.corrcoef(one - neutral, other - neutral)[0, 1] < -0.5, (
        "opposite histories produced the same taste direction"
    )


def test_factorisation_falls_back_for_a_history_it_cannot_use():
    ratings = make_ratings()
    model = BiasedMF(n_factors=4, min_item_ratings=3).fit_global(ratings)
    scores = model.score_user(np.array([10**9]), np.array([5.0]), ratings["movieId"].unique()[:5])
    assert np.isfinite(scores).all()


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def test_evaluation_reports_each_model_at_each_history_length():
    ratings = make_ratings()
    split = split_users(ratings, n_test_users=10, min_ratings=40, seed=0)
    per_user = evaluate_generalisation(
        [UserMean(), ItemMean(), BiasModel()], ratings, split, given_sizes=(10, 25)
    )
    report = summarise(per_user)
    assert set(report["model"]) == {"user_mean", "item_mean", "bias"}
    assert set(report["n_given"]) == {10, 25}
    assert (report["users"] > 0).all()


def test_summary_averages_per_user_not_per_rating():
    """Otherwise a few very active users decide the headline number."""
    per_user = pd.DataFrame(
        {
            "model": ["m"] * 2,
            "n_given": [10, 10],
            "userId": [1, 2],
            "n": [1000, 10],  # wildly different activity
            "rmse": [1.0, 0.0],
            "mae": [1.0, 0.0],
        }
    )
    assert summarise(per_user).loc[0, "rmse"] == pytest.approx(0.5)


def test_user_bias_is_learned_from_the_history():
    """The term whose absence made an earlier factorisation lose to plain bias.

    A harsh rater must be predicted lower than a generous one, from history
    alone, with no refitting of anything global.
    """
    ratings = make_ratings(n_users=80, n_items=90, seed=2)
    model = BiasedMF(n_factors=8, min_item_ratings=3).fit_global(ratings)
    items = ratings["movieId"].unique()[:20]
    neutral = model._baseline(items)

    harsh = model.score_user(items, neutral - 1.0, items)
    generous = model.score_user(items, neutral + 1.0, items)
    assert (generous > harsh).all()
    assert generous.mean() > neutral.mean() > harsh.mean()


def test_user_bias_is_shrunk_for_a_short_history():
    """One extreme rating must not buy a full-size personal offset."""
    ratings = make_ratings(n_users=80, n_items=90, seed=3)
    model = BiasedMF(n_factors=8, min_item_ratings=3, user_prior=10.0).fit_global(ratings)
    items = ratings["movieId"].unique()[:30]
    neutral = model._baseline(items)

    one_rating = model.score_user(items[:1], neutral[:1] + 2.0, items)
    many_ratings = model.score_user(items, neutral + 2.0, items)
    assert (many_ratings > one_rating).all(), "a long history should move the offset further"


# --------------------------------------------------------------------------
# the content tower
# --------------------------------------------------------------------------
def make_item_features(n_items: int = 90, n_dims: int = 6, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (n_items, n_dims)).astype("float32"),
        index=pd.Index(range(n_items), name="movieId"),
        columns=[f"d{i}" for i in range(n_dims)],
    )


def test_content_tower_scores_films_no_one_has_rated():
    """The whole reason it exists: MovieLens covers 84k of 1.15M films."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=60, n_items=50, seed=5)
    features = make_item_features(n_items=90)  # 40 films nobody has rated
    model = ContentTower(features, learn_weights=False).fit_global(ratings)

    unrated = np.arange(50, 90)
    scores = model.score_user(np.array([1, 2, 3]), np.array([5.0, 4.0, 2.0]), unrated)
    assert np.isfinite(scores).all()
    assert scores.std() > 0, "every unrated film got the same score"


def test_content_profile_reflects_what_the_user_liked():
    """Two opposite histories must produce opposite content profiles."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=60, n_items=90, seed=6)
    features = make_item_features(n_items=90)
    model = ContentTower(features, learn_weights=False).fit_global(ratings)

    items = np.arange(40)
    baseline = model._baseline(items)
    liked_first = model.score_user(
        items, np.concatenate([baseline[:20] + 1.5, baseline[20:] - 1.5]), items
    )
    liked_second = model.score_user(
        items, np.concatenate([baseline[:20] - 1.5, baseline[20:] + 1.5]), items
    )
    deviation_one = liked_first - baseline
    deviation_two = liked_second - baseline
    assert np.corrcoef(deviation_one, deviation_two)[0, 1] < -0.5


def test_content_profile_ignores_a_user_who_rates_everything_the_same():
    """Weighting by rating rather than by surprise would make watching == liking."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=60, n_items=90, seed=7)
    features = make_item_features(n_items=90)
    model = ContentTower(features, learn_weights=False).fit_global(ratings)

    items = np.arange(30)
    flat = model._baseline(items)  # rates every film exactly as expected
    profile, _ = model._profile(items, flat)
    assert np.abs(profile).max() < 1e-5


def test_content_tower_handles_films_it_has_no_features_for():
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=60, n_items=90, seed=8)
    model = ContentTower(make_item_features(n_items=50), learn_weights=False).fit_global(ratings)
    scores = model.score_user(np.array([10**9]), np.array([5.0]), np.array([1, 10**9]))
    assert np.isfinite(scores).all()


def test_learned_weights_are_global_not_per_user():
    """Weights are fitted once across users; scoring must not change them."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=90, seed=9)
    features = make_item_features(n_items=90)
    model = ContentTower(features, learn_weights=True, max_fit_users=60).fit_global(ratings)

    before = model.weights_.copy()
    model.score_user(np.arange(20), np.full(20, 4.0), np.arange(30))
    assert np.array_equal(before, model.weights_)
