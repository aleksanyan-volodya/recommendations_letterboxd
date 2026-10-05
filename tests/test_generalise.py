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


# --------------------------------------------------------------------------
# item-bias strategies and the cold-item protocol
# --------------------------------------------------------------------------
def test_unknown_item_bias_strategies_disagree_about_unrated_films():
    """The three strategies must actually be three different models.

    ``known`` pins an unrated film to the global mean, ``none`` refuses the term
    for every film, and ``predicted`` infers one from content. The differences
    only show on a candidate set that mixes rated and unrated films, which is
    why the cold-item protocol scores them together.
    """
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=60, seed=11)
    features = make_item_features(n_items=120, n_dims=6)  # 60 films nobody rated
    targets = np.arange(120)  # rated and unrated competing in one ranking
    unrated = targets >= 60
    given_items, given_values = np.arange(10), np.full(10, 4.0)

    scored = {}
    for strategy in ("known", "none", "predicted"):
        model = ContentTower(
            features, item_bias=strategy, learn_weights=False, bias_reg=0.1
        ).fit_global(ratings)
        scored[strategy] = model.score_user(given_items, given_values, targets)

    for left, right in (("known", "none"), ("known", "predicted"), ("none", "predicted")):
        assert not np.allclose(scored[left], scored[right]), f"{left} and {right} agree"

    # `known` gives every unrated film the same b_i (zero); `predicted` must not.
    assert np.allclose(scored["known"][unrated], scored["none"][unrated])
    assert (scored["predicted"][unrated] - scored["none"][unrated]).std() > 0


def test_missing_bias_strategies_cannot_differ_on_cold_films_alone():
    """The finding that shaped the protocol, pinned so it cannot be forgotten.

    On a candidate set of *only* crowd-less films, `known` and `none` are the
    same model. An experiment that scored cold films in isolation would compare
    them and correctly report no difference, which would be a true number and a
    false conclusion.
    """
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=60, seed=17)
    features = make_item_features(n_items=120, n_dims=6)
    unrated = np.arange(60, 120)
    history, values = np.arange(10), np.full(10, 4.0)

    scores = [
        ContentTower(features, item_bias=strategy, learn_weights=False)
        .fit_global(ratings)
        .score_user(history, values, unrated)
        for strategy in ("known", "none")
    ]
    assert np.allclose(*scores)


def test_predicted_item_bias_is_learned_from_rated_films_only():
    """Fitting it on unrated films would be fitting on zeros it invented."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=60, seed=12)
    features = make_item_features(n_items=120, n_dims=6)
    model = ContentTower(features, item_bias="predicted", learn_weights=False).fit_global(ratings)

    assert model.bias_weights_ is not None
    # A film the crowd rated keeps its measured bias, never the predicted one.
    rated = np.array([3])
    measured = model.global_mean_ + float(model.item_bias_.loc[3])
    assert model._baseline(rated)[0] == pytest.approx(measured, abs=1e-5)


def test_item_bias_none_ignores_crowd_quality_when_ranking_candidates():
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=80, n_items=60, seed=13)
    model = ContentTower(
        make_item_features(n_items=60), item_bias="none", learn_weights=False
    ).fit_global(ratings)
    assert np.allclose(model._baseline(np.arange(60)), model.global_mean_)


def test_item_bias_none_still_uses_the_crowd_to_read_the_history():
    """Withholding `b_i` is a ranking decision, not a profile decision.

    Without it, a user who rates an acclaimed film 5 looks enthusiastic rather
    than ordinary, and their taste vector drifts toward whatever is merely good.
    """
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=80, n_items=60, seed=18)
    model = ContentTower(
        make_item_features(n_items=60), item_bias="none", learn_weights=False
    ).fit_global(ratings)

    history = np.arange(30)
    assert not np.allclose(model._history_baseline(history), model.global_mean_)
    expected = model.global_mean_ + model.item_bias_.loc[history].to_numpy()
    assert model._history_baseline(history) == pytest.approx(expected, abs=1e-5)


def test_unknown_item_bias_strategy_is_rejected():
    from lbrec.generalise import ContentTower

    with pytest.raises(ValueError, match="item_bias"):
        ContentTower(make_item_features(n_items=10), item_bias="popularity")


def borrowed_tower(seed: int = 21, **kwargs):
    """A tower whose second crowd rates every film exactly twice as sharply.

    Films 0-59 are rated by the training crowd; films 60-119 only by the second
    one, which gives them a bias of 0.4 (i.e. 0.2 on the training crowd's scale).
    """
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=60, seed=seed)
    features = make_item_features(n_items=130, n_dims=6)
    own = ContentTower(features, learn_weights=False).fit_global(ratings).item_bias_
    lent = pd.concat([2.0 * own.astype("float64"), pd.Series(0.4, index=range(60, 120))])
    model = ContentTower(
        features, item_bias="borrowed", borrowed_bias=lent, learn_weights=False, **kwargs
    ).fit_global(ratings)
    return model


def test_borrowed_bias_is_calibrated_onto_the_training_crowds_scale():
    """Uncalibrated, a sharper crowd's biases would win on spread alone."""
    model = borrowed_tower()
    intercept, slope = model.borrow_map_
    assert slope == pytest.approx(0.5, abs=1e-4)
    assert intercept == pytest.approx(0.0, abs=1e-4)
    assert model._baseline(np.array([100]))[0] == pytest.approx(model.global_mean_ + 0.2)


def test_a_borrowed_bias_never_overrides_a_measured_one():
    model = borrowed_tower()
    measured = model.global_mean_ + float(model.item_bias_.loc[3])
    assert model._baseline(np.array([3]))[0] == pytest.approx(measured, abs=1e-5)


def test_films_neither_crowd_rated_fall_back_to_the_content_guess():
    from lbrec.generalise import ContentTower

    borrowed = borrowed_tower()
    guessed = ContentTower(
        borrowed.features_, item_bias="predicted", learn_weights=False
    ).fit_global(make_ratings(n_users=120, n_items=60, seed=21))
    orphans = np.arange(120, 130)  # in the features, rated by nobody
    assert borrowed._baseline(orphans) == pytest.approx(guessed._baseline(orphans), abs=1e-5)


def test_a_borrowed_bias_also_reads_the_history():
    """A film only the second crowd knows is still a measured film when reading taste."""
    model = borrowed_tower()
    assert model._history_baseline(np.array([100]))[0] == pytest.approx(model.global_mean_ + 0.2)


def test_borrowed_strategy_needs_a_second_crowd():
    from lbrec.generalise import ContentTower

    with pytest.raises(ValueError, match="borrowed_bias"):
        ContentTower(make_item_features(n_items=10), item_bias="borrowed")


def test_cold_items_are_hidden_from_the_global_fit():
    """The point of the protocol: the model must not know the cold films.

    If a cold film's ratings reached the item-bias table, the evaluation would
    measure memorisation of exactly the films it claims are unknown.
    """
    from lbrec.generalise import ContentTower, cold_item_split, evaluate_cold_items

    ratings = make_ratings(n_users=120, n_items=90, seed=14)
    split = split_users(ratings, n_test_users=20, min_ratings=20, seed=0)
    cold = cold_item_split(ratings, share=0.2, seed=0)
    features = make_item_features(n_items=90, n_dims=6)

    model = ContentTower(features, learn_weights=False)
    per_user = evaluate_cold_items([model], ratings, split, cold, given_sizes=(10,), seed=0)

    assert not per_user.empty
    assert not set(model.item_bias_.index) & set(cold.tolist())


def make_catalogue(n_films: int = 80, seed: int = 3) -> pd.DataFrame:
    """Catalogue rows with every field the feature builder reads."""
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(30)]
    pick = lambda k: list(rng.choice(words, size=k, replace=False))  # noqa: E731
    return pd.DataFrame(
        {
            "tmdb_id": np.arange(1000, 1000 + n_films),
            "genres": [pick(2) for _ in range(n_films)],
            "keywords": [pick(4) for _ in range(n_films)],
            "directors": [pick(1) for _ in range(n_films)],
            "cast": [pick(3) for _ in range(n_films)],
            "production_countries": [pick(1) for _ in range(n_films)],
            "year": rng.integers(1950, 2020, n_films),
            "runtime": rng.integers(70, 180, n_films),
            "vote_count": rng.integers(0, 5000, n_films),
            "vote_average": rng.uniform(4, 8, n_films),
            "original_language": rng.choice(["en", "fr", "ja"], n_films),
            "overview": [" ".join(pick(8)) for _ in range(n_films)],
        }
    )


def test_catalogue_vectors_are_keyed_by_tmdb_id_one_row_per_film():
    from lbrec.generalise import catalogue_item_features

    films = make_catalogue()
    vectors = catalogue_item_features(pd.concat([films, films.head(5)]), n_components=8)
    assert list(vectors.index) == films["tmdb_id"].tolist()
    assert vectors.shape[1] == 8
    assert np.isfinite(vectors.to_numpy()).all()


def test_fitting_on_a_sample_still_embeds_every_film():
    """The memory fix must not drop or blank the films outside the sample."""
    from lbrec.generalise import catalogue_item_features

    films = make_catalogue(n_films=80)
    sampled = catalogue_item_features(films, n_components=8, fit_rows=40)
    assert len(sampled) == 80
    assert (np.abs(sampled.to_numpy()).sum(axis=1) > 0).all()


def test_a_sample_as_large_as_the_data_is_the_full_fit():
    from lbrec.generalise import catalogue_item_features

    films = make_catalogue(n_films=60)
    full = catalogue_item_features(films, n_components=8, fit_rows=60)
    larger = catalogue_item_features(films, n_components=8, fit_rows=10_000)
    assert np.allclose(full.to_numpy(), larger.to_numpy(), atol=1e-5)


def test_a_measured_second_crowd_beats_a_guess_on_the_films_it_lends():
    """The experiment in miniature, run through the real protocol.

    The second crowd is the training users again, but *with* the cold films and
    on a sharper scale -- a crowd that measured what the first one never saw.
    Its calibrated bias must predict the hidden films better than a guess from
    metadata that, here, carries no information about quality at all.
    """
    from lbrec.generalise import (
        ContentTower,
        StratifiedRanker,
        cold_item_split,
        evaluate_cold_items,
    )

    ratings = make_ratings(n_users=160, n_items=90, seed=19)
    split = split_users(ratings, n_test_users=30, min_ratings=20, seed=0)
    cold = cold_item_split(ratings, share=0.25, seed=0)
    features = make_item_features(n_items=90, n_dims=6)

    second = ratings[ratings["userId"].isin(set(split.train_users.tolist()))]
    stats = second.groupby("movieId")["rating"].agg(["sum", "count"])
    lent = 2.0 * (stats["sum"] - stats["count"] * second["rating"].mean()) / (stats["count"] + 20)

    models = [
        ContentTower(features, item_bias="predicted", learn_weights=False),
        ContentTower(features, item_bias="borrowed", borrowed_bias=lent, learn_weights=False),
        StratifiedRanker(
            ContentTower(features, item_bias="borrowed", borrowed_bias=lent, learn_weights=False)
        ),
    ]
    per_user = evaluate_cold_items(models, ratings, split, cold, given_sizes=(10,), seed=0)
    cold_rmse = per_user.groupby("model")["rmse_cold"].mean()

    assert set(cold_rmse.index) == {
        "tower_predicted",
        "tower_borrowed",
        "stratified_tower_borrowed",
    }
    assert cold_rmse["tower_borrowed"] < cold_rmse["tower_predicted"]


def test_cold_item_scoring_never_shows_a_cold_film_in_the_history():
    """A user's own rating of a cold film would leak the answer into the input."""
    from lbrec.generalise import cold_item_split, evaluate_cold_items

    ratings = make_ratings(n_users=120, n_items=90, seed=15)
    split = split_users(ratings, n_test_users=20, min_ratings=20, seed=0)
    cold = set(cold_item_split(ratings, share=0.3, seed=0).tolist())

    seen_histories: list[np.ndarray] = []

    class Spy:
        name = "spy"

        def fit_global(self, ratings):
            return self

        def score_user(self, given_items, given_ratings, target_items):
            seen_histories.append(np.asarray(given_items))
            return np.full(len(target_items), 3.5)

    evaluate_cold_items([Spy()], ratings, split, np.array(sorted(cold)), given_sizes=(10,), seed=0)

    assert seen_histories
    for history in seen_histories:
        assert not set(history.tolist()) & cold


def test_cold_item_split_only_picks_films_with_enough_ratings():
    """A cold film with two ratings gives an RMSE that is mostly noise."""
    from lbrec.generalise import cold_item_split

    ratings = make_ratings(n_users=60, n_items=200, seed=16)
    support = ratings["movieId"].value_counts()
    cold = cold_item_split(ratings, share=0.5, seed=0, min_ratings=5)
    assert (support.loc[cold] >= 5).all()


def test_tail_share_detects_a_model_that_suppresses_crowd_less_films():
    """The fairness metric must react to the bias it is there to catch.

    A model that adds a bonus to every well-known film should fill the top of the
    list with them, pushing `tail_share` below what the user's own ratings say it
    should be.
    """
    from lbrec.generalise import _ranking_metrics

    is_cold = np.array([True] * 10 + [False] * 10)
    actual = np.array([5.0] * 10 + [2.0] * 10)  # the user loved the cold films

    fair = _ranking_metrics(actual, actual.copy(), is_cold, k=10)
    assert fair["tail_share"] == pytest.approx(1.0)
    assert fair["tail_share_actual"] == pytest.approx(1.0)

    famous_bonus = actual + np.where(is_cold, 0.0, 4.0)  # enough to invert the order
    biased = _ranking_metrics(actual, famous_bonus, is_cold, k=10)
    assert biased["tail_share"] == pytest.approx(0.0)
    assert biased["tail_share_actual"] == pytest.approx(1.0)


def test_top_k_ties_are_not_broken_in_favour_of_older_films():
    """MovieLens rows arrive movieId-ordered, and movieId tracks release year.

    With every prediction identical, `argsort` alone would hand the whole top ten
    to whichever films happen to come first -- reporting a confident tail share
    that is really just row order.
    """
    from lbrec.generalise import _ranking_metrics

    is_cold = np.array([True] * 50 + [False] * 50)
    flat = np.full(100, 3.5)
    share = _ranking_metrics(flat, flat, is_cold, k=10)["tail_share"]
    assert 0.0 < share < 1.0


def test_bias_weight_traces_the_frontier_between_the_two_corners():
    """`bias_weight` must interpolate, with its ends matching the named strategies.

    At 0 the crowd term is gone, so the ranking must match `item_bias="none"`;
    at 1 nothing is discounted. In between the crowd's influence shrinks
    monotonically, which is what makes the curve a usable knob rather than a
    third arbitrary setting.
    """
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=120, n_items=60, seed=19)
    features = make_item_features(n_items=120, n_dims=6)
    targets = np.arange(120)
    history, values = np.arange(10), np.full(10, 4.0)

    def scored(**kwargs):
        model = ContentTower(features, learn_weights=False, **kwargs).fit_global(ratings)
        return model.score_user(history, values, targets)

    fair = scored(item_bias="none")
    damped_out = scored(item_bias="predicted", bias_weight=0.0)
    assert np.allclose(fair, damped_out)

    # The crowd's contribution must shrink as the weight falls.
    full = scored(item_bias="predicted", bias_weight=1.0)
    half = scored(item_bias="predicted", bias_weight=0.5)
    crowd_full = np.abs(full - fair).mean()
    crowd_half = np.abs(half - fair).mean()
    assert crowd_half < crowd_full
    assert crowd_half == pytest.approx(crowd_full / 2, rel=1e-5)


def test_bias_weight_shows_up_in_the_model_name():
    """Sweep rows are useless if every point reports as the same model."""
    from lbrec.generalise import ContentTower

    features = make_item_features(n_items=20)
    assert ContentTower(features, item_bias="predicted").name == "tower_predicted"
    assert (
        ContentTower(features, item_bias="predicted", bias_weight=0.25).name
        == "tower_predicted_b0.25"
    )


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------
def _frontier_rows(rows: list[dict]) -> pd.DataFrame:
    """Per-user records shaped like the cold-item protocol's output."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "n_given": 25,
                "userId": user,
                "spearman": spearman,
                "tail_share": tail,
                "tail_share_actual": 0.30,
            }
            for model, spearman, tail in rows
            for user in range(20)
        ]
    )


def test_objective_rejects_an_accurate_model_that_suppresses_the_tail():
    """The constraint comes first: ranking well is not a licence to hide films."""
    from lbrec.generalise import frontier

    per_user = _frontier_rows(
        [("suppressor", 0.40, 0.05), ("fair_but_weak", 0.10, 0.30), ("fair_and_good", 0.30, 0.29)]
    )
    verdict = frontier(per_user)
    picked = verdict[verdict["best"]]["model"].tolist()
    assert picked == ["fair_and_good"]
    assert not verdict.set_index("model").loc["suppressor", "fair"]


def test_objective_prefers_the_best_ranking_among_the_fair_models():
    from lbrec.generalise import frontier

    per_user = _frontier_rows([("fair_weak", 0.12, 0.30), ("fair_strong", 0.25, 0.28)])
    verdict = frontier(per_user)
    assert verdict[verdict["best"]]["model"].tolist() == ["fair_strong"]


def test_objective_reports_overshooting_without_rewarding_it():
    """Showing obscure films more than they deserve is a different product.

    It is not disqualified -- it clears the bar -- but it must not beat a model
    that is both fair and ranks better, and the fairness ratio has to make the
    overshoot visible rather than hide it at 1.0.
    """
    from lbrec.generalise import frontier

    per_user = _frontier_rows([("overshooter", 0.15, 0.60), ("fair_and_good", 0.30, 0.29)])
    verdict = frontier(per_user).set_index("model")
    assert verdict.loc["overshooter", "fair"]
    assert verdict.loc["overshooter", "fairness"] == pytest.approx(2.0)
    assert not verdict.loc["overshooter", "best"]


def test_objective_marks_nothing_when_no_model_is_fair():
    """Silence beats a false winner: it is the signal to change architecture."""
    from lbrec.generalise import frontier

    verdict = frontier(_frontier_rows([("a", 0.40, 0.05), ("b", 0.35, 0.02)]))
    assert not verdict["best"].any()
    assert not verdict["fair"].any()


def test_objective_needs_the_cold_item_columns():
    from lbrec.generalise import frontier

    with pytest.raises(ValueError, match="tail_share"):
        frontier(pd.DataFrame({"model": ["x"], "n_given": [10], "spearman": [0.2]}))


# --------------------------------------------------------------------------
# stratified ranking
# --------------------------------------------------------------------------
def test_interleaving_holds_each_group_at_its_own_rate():
    """The fairness guarantee, by construction rather than by tuning."""
    from lbrec.generalise import _interleave

    # 30 crowd-less films and 70 known ones; the known ones score far higher, so
    # any direct comparison would bury the tail completely.
    known = np.array([False] * 30 + [True] * 70)
    scores = np.where(known, 5.0, 2.0) + np.linspace(0, 0.5, 100)

    ranked = _interleave(scores, known)
    top_ten = np.argsort(-ranked)[:10]
    assert 0.2 <= (~known[top_ten]).mean() <= 0.4, "top ten did not match the 30% base rate"


def test_interleaving_keeps_the_full_model_order_inside_each_group():
    """Fairness must not cost ranking power where the comparison is valid."""
    from lbrec.generalise import _interleave

    known = np.array([False, False, False, True, True, True])
    scores = np.array([1.0, 3.0, 2.0, 10.0, 30.0, 20.0])
    ranked = _interleave(scores, known)

    for member in (known, ~known):
        index = np.flatnonzero(member)
        by_score = index[np.argsort(-scores[index])]
        by_rank = index[np.argsort(-ranked[index])]
        assert list(by_score) == list(by_rank)


def test_interleaving_survives_a_group_being_empty():
    from lbrec.generalise import _interleave

    scores = np.array([1.0, 3.0, 2.0])
    ranked = _interleave(scores, np.array([True, True, True]))
    assert list(np.argsort(-ranked)) == list(np.argsort(-scores))


def test_stratified_ranker_still_predicts_calibrated_ratings():
    """Ranking by an interleaved position must not corrupt the prediction.

    RMSE has to stay comparable with the underlying tower, or the fairness fix
    would look like an accuracy regression that it is not.
    """
    from lbrec.generalise import ContentTower, StratifiedRanker

    ratings = make_ratings(n_users=120, n_items=60, seed=20)
    features = make_item_features(n_items=120, n_dims=6)
    tower = ContentTower(features, item_bias="predicted", learn_weights=False)
    model = StratifiedRanker(tower).fit_global(ratings)

    history, values, targets = np.arange(10), np.full(10, 4.0), np.arange(120)
    assert np.allclose(
        model.score_user(history, values, targets),
        tower.score_user(history, values, targets),
    )
    ranked = model.rank_user(history, values, targets)
    assert not np.allclose(ranked, model.score_user(history, values, targets))


def test_crowd_knowledge_is_read_from_the_item_table_not_a_flag():
    """It must work for a stranger, where no experiment marks the cold films."""
    from lbrec.generalise import ContentTower

    ratings = make_ratings(n_users=80, n_items=40, seed=21)
    model = ContentTower(make_item_features(n_items=90), learn_weights=False).fit_global(ratings)
    known = model.knows_crowd_opinion(np.arange(90))
    assert known[:40].all()
    assert not known[40:].any()
