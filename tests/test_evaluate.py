"""Tests for the evaluation harness and the baselines.

The harness is the thing that decides whether any later model is an improvement,
so its own correctness matters more than any model's score. The properties
pinned here are the ones whose failure would silently inflate every result:
no fold sees its own test rows, transformers are refitted per fold, and metrics
are sliced by popularity rather than averaged into one flattering number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbrec.catalogue import popularity_band
from lbrec.evaluate import cross_validate, evaluate, tail_summary
from lbrec.features import (
    CROWD_PRIOR_MEAN,
    build_film_features,
    neutralise_popularity,
    shrink_crowd_score,
)
from lbrec.models import ContentRidge, CrowdScore, GlobalMean


def make_catalogue(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genres = ["Drama", "Comedy", "Horror", "Documentary"]
    return pd.DataFrame(
        {
            "tmdb_id": pd.Series(range(n), dtype="Int64"),
            "genres": [[genres[i % 4]] for i in range(n)],
            "keywords": [[f"kw{i % 17}", f"kw{i % 5}"] for i in range(n)],
            "directors": [[f"dir_{i % 13}"] for i in range(n)],
            "cast": [[f"actor_{i % 23}", f"actor_{(i + 1) % 23}"] for i in range(n)],
            "production_countries": [["FR" if i % 3 else "US"] for i in range(n)],
            "year": pd.Series(1950 + (np.arange(n) % 70), dtype="Int64"),
            "runtime": pd.Series(rng.integers(70, 200, n), dtype="Int64"),
            "vote_count": pd.Series(rng.integers(1, 40_000, n), dtype="Int64"),
            "vote_average": pd.Series(rng.uniform(4, 9, n), dtype="Float64"),
            "original_language": ["en" if i % 2 else "fr" for i in range(n)],
            "overview": [
                f"A {genres[i % 4].lower()} about theme{i % 11} and theme{i % 7} set in a city."
                for i in range(n)
            ],
            "tagline": ["" for _ in range(n)],
        }
    )


@pytest.fixture
def dataset():
    catalogue = make_catalogue()
    features = build_film_features(catalogue)
    rng = np.random.default_rng(1)
    ratings = pd.Series(np.clip(rng.normal(3.2, 0.9, len(catalogue)), 1, 5))
    return features, ratings, catalogue


def test_global_mean_predicts_the_training_mean():
    features = build_film_features(make_catalogue(20))
    ratings = pd.Series([2.0] * 10 + [4.0] * 10)
    model = GlobalMean().fit(features, ratings)
    assert model.predict(features) == pytest.approx(np.full(20, 3.0))


def test_crowd_score_learns_the_rescaling_rather_than_assuming_it():
    """A 0-10 crowd score does not map onto a 1-5 personal scale by division."""
    catalogue = make_catalogue(60)
    features = build_film_features(catalogue)
    crowd = features["crowd_shrunk"].astype(float)
    ratings = 0.5 * crowd - 0.5  # an exact linear relationship
    model = CrowdScore().fit(features, ratings)
    assert model.slope_ == pytest.approx(0.5, abs=1e-6)
    assert model.predict(features) == pytest.approx(ratings.to_numpy(), abs=1e-6)


def test_a_single_ten_out_of_ten_vote_is_shrunk_to_the_prior():
    """The bug the first real recommendation run exposed.

    3,093 catalogue films have exactly one vote of 10/10. Ranking on the raw
    average put them at the top of the list ahead of everything else.
    """
    shrunk = shrink_crowd_score(pd.Series([10.0]), pd.Series([1]))
    assert shrunk.iloc[0] < CROWD_PRIOR_MEAN + 0.1
    # A genuinely well-liked film with real support keeps its score.
    well_supported = shrink_crowd_score(pd.Series([8.0]), pd.Series([20_000]))
    assert well_supported.iloc[0] == pytest.approx(8.0, abs=0.01)


def test_no_votes_means_unknown_not_terrible():
    """TMDb records vote_average as 0.0 for unrated films; 87k of them exist."""
    shrunk = shrink_crowd_score(pd.Series([0.0, np.nan]), pd.Series([0, 0]))
    assert shrunk.tolist() == pytest.approx([CROWD_PRIOR_MEAN, CROWD_PRIOR_MEAN])


def test_shrinkage_is_monotone_in_support():
    """More votes behind the same average means more of that average survives."""
    scores = shrink_crowd_score(pd.Series([9.0] * 4), pd.Series([1, 10, 100, 10_000]))
    assert scores.is_monotonic_increasing
    assert scores.iloc[-1] > scores.iloc[0]


def test_crowd_score_falls_back_when_a_film_has_no_crowd_score():
    catalogue = make_catalogue(30)
    catalogue.loc[0, "vote_average"] = pd.NA
    features = build_film_features(catalogue)
    ratings = pd.Series(np.linspace(1, 5, 30))
    predictions = CrowdScore().fit(features, ratings).predict(features)
    assert np.isfinite(predictions).all()


def test_content_ridge_learns_a_signal_it_can_see(dataset):
    """A rating driven entirely by runtime must be recoverable."""
    features, _, catalogue = dataset
    ratings = pd.Series(
        np.interp(catalogue["runtime"].astype(float), (70, 200), (1.0, 5.0)), dtype=float
    )
    model = ContentRidge(keyword_components=8).fit(features, ratings)
    predicted = model.predict(features)
    assert np.corrcoef(predicted, ratings)[0, 1] > 0.8


def test_neutralise_popularity_changes_only_popularity(dataset):
    features, _, _ = dataset
    neutral = neutralise_popularity(features, reference=features)
    assert neutral["log_votes"].nunique() == 1
    for column in ("genres", "keywords", "year", "runtime", "vote_average"):
        pd.testing.assert_series_equal(neutral[column], features[column])


def test_debiased_model_ignores_popularity_at_prediction_time():
    """Two films differing only in popularity must score identically once debiased."""
    catalogue = make_catalogue(80)
    features = build_film_features(catalogue)
    ratings = pd.Series(np.linspace(1, 5, 80))
    model = ContentRidge(debias=True, keyword_components=8).fit(features, ratings)

    probe = features.iloc[[0, 0]].copy().reset_index(drop=True)
    probe.loc[1, "log_votes"] = probe.loc[0, "log_votes"] + 5.0
    predictions = model.predict(probe)
    assert predictions[0] == pytest.approx(predictions[1])


def test_cross_validation_predicts_every_row_exactly_once(dataset):
    features, ratings, _ = dataset
    predictions = cross_validate([GlobalMean()], features, ratings, folds=5)
    assert len(predictions) == len(features)
    assert sorted(predictions["row"]) == sorted(features.index)


def test_each_row_is_predicted_by_a_model_that_never_saw_it(dataset):
    """The guard against the harness flattering itself."""
    features, ratings, _ = dataset

    class LeakDetector:
        name = "leak_detector"

        def fit(self, features, ratings):
            self.seen_ = set(features.index)
            return self

        def predict(self, features):
            assert self.seen_.isdisjoint(features.index), "model was asked to score a training row"
            return np.zeros(len(features))

    cross_validate([LeakDetector()], features, ratings, folds=5)


def test_popularity_bands_are_absolute_not_quantiles():
    """Bands must mean the same thing regardless of what set they are applied to.

    Quantiles cut against a user's own library made "decile 1" mean "the least
    popular film I have watched" -- which in the real catalogue is the top ~10%
    of cinema. Absolute vote thresholds cannot drift like that.
    """
    votes = pd.Series([0, 3, 12, 50, 250, 1500, 5000, 50_000])
    assert popularity_band(votes).tolist() == [
        "0",
        "1-4",
        "5-19",
        "20-99",
        "100-499",
        "500-2k",
        "2k-10k",
        "10k+",
    ]
    # Same inputs inside a much more popular set must land in the same bands.
    bigger = pd.Series([0, 3, 12, 50, 250, 1500, 5000, 50_000] + [90_000] * 500)
    assert popularity_band(bigger).head(8).tolist() == popularity_band(votes).tolist()


def test_missing_vote_counts_are_treated_as_zero():
    assert popularity_band(pd.Series([None, pd.NA])).tolist() == ["0", "0"]


def test_evaluation_reports_every_model_overall_and_per_decile(dataset):
    features, ratings, catalogue = dataset
    result = evaluate(
        [GlobalMean(), CrowdScore()], features, ratings, catalogue["vote_count"], folds=3
    )
    assert set(result.overall["model"]) == {"global_mean", "crowd_score"}
    assert set(result.by_decile["model"]) == {"global_mean", "crowd_score"}
    assert result.by_decile["band"].nunique() > 1
    assert (result.overall["n"] == len(features)).all()


def test_global_mean_rmse_matches_the_rating_spread(dataset):
    """The floor is the standard deviation, so the harness must reproduce it."""
    features, ratings, catalogue = dataset
    result = evaluate([GlobalMean()], features, ratings, catalogue["vote_count"], folds=5)
    rmse = float(result.overall.loc[0, "rmse"])
    assert rmse == pytest.approx(float(ratings.std(ddof=0)), rel=0.1)


def test_tail_summary_exposes_a_head_only_model(dataset):
    """A model good on famous films and bad on obscure ones must show a positive gap."""
    by_decile = pd.DataFrame(
        {
            "model": ["m"] * 8,
            "band": ["0", "1-4", "5-19", "20-99", "100-499", "500-2k", "2k-10k", "10k+"],
            "rmse": [1.5, 1.4, 1.3, 1.4, 0.5, 0.5, 0.5, 0.5],
        }
    )
    summary = tail_summary(by_decile).set_index("model")
    assert summary.loc["m", "rmse_tail"] == pytest.approx(1.4)  # mean of the four tail bands
    assert summary.loc["m", "rmse_head"] == pytest.approx(0.5)
    assert summary.loc["m", "gap"] > 0


def test_vectoriser_settings_survive_sklearn_cloning():
    """Regression: options behind ``**kwargs`` are invisible to ``get_params``.

    ``ColumnTransformer`` clones its transformers before fitting, so any
    ``__init__`` argument not named explicitly is silently dropped. That turned a
    3,000-term vocabulary cap into 86,133 columns on the full catalogue and a
    161 GB allocation -- with no error until the memory ran out.
    """
    from sklearn.base import clone

    from lbrec.features import TextBlock

    block = TextBlock(components=8, max_features=1500, min_df=2, token_pattern=r"[^ ]+")
    copied = clone(block)
    assert copied.max_features == 1500
    assert copied.min_df == 2
    assert copied.token_pattern == r"[^ ]+"
    assert copied.components == 8


def test_vocabulary_cap_is_actually_enforced():
    """The cap must bind on real data, not merely be stored on the object."""
    from sklearn.base import clone

    from lbrec.features import TextBlock

    documents = [f"tok{i} tok{i + 1} shared" for i in range(500)]
    fitted = clone(TextBlock(max_features=50, min_df=1, token_pattern=r"[^ ]+")).fit(documents)
    assert fitted.transform(documents).shape[1] <= 50
