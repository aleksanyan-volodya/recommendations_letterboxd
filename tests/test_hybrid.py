"""Tests for the collaborative leg and the blend.

Both of these shipped with a bug that produced plausible-looking output rather
than an error, so each has a regression test here:

* factorising item-centred ratings removes how well a film is rated, which is
  the strongest collaborative signal there is;
* fitting blend weights on in-sample predictions hands everything to whichever
  leg overfits hardest, silently collapsing the blend to one model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbrec.models import CollaborativeFold, ContentRidge, SupportGatedBlend
from tests.test_evaluate import make_catalogue


def make_factors(tmdb_ids, *, n_factors: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tmdb_ids)
    frame = pd.DataFrame(
        rng.normal(size=(n, n_factors)).astype("float32"),
        columns=[f"f{i}" for i in range(n_factors)],
    )
    frame.insert(0, "movieId", np.arange(n, dtype="int32"))
    frame.insert(1, "item_mean", rng.uniform(2.5, 4.5, n).astype("float32"))
    frame.insert(2, "n_ratings", rng.integers(20, 5000, n))
    return frame


def make_links(tmdb_ids) -> pd.DataFrame:
    return pd.DataFrame({"movieId": np.arange(len(tmdb_ids), dtype="int32"), "tmdbId": tmdb_ids})


@pytest.fixture
def collaborative_setup():
    catalogue = make_catalogue(80)
    from lbrec.features import build_film_features

    features = build_film_features(catalogue)
    covered = catalogue["tmdb_id"].astype("int64").to_numpy()[:60]  # 20 films uncovered
    return features, make_factors(covered), make_links(covered)


def test_item_mean_is_part_of_the_representation(collaborative_setup):
    """The regression test for centring away the quality signal."""
    features, factors, links = collaborative_setup
    model = CollaborativeFold(factors, links)
    assert "item_mean" in model.columns

    # A user who simply agrees with the crowd must be perfectly learnable.
    means = factors.set_index("movieId")["item_mean"]
    ids = pd.to_numeric(features["tmdb_id"], errors="coerce")
    ratings = pd.Series(ids.map(dict(zip(links["tmdbId"], means, strict=False))).fillna(3.0))

    fitted = model.fit(features, ratings)
    covered = fitted.coverage(features)
    predicted = fitted.predict(features)
    assert np.corrcoef(predicted[covered], ratings[covered])[0, 1] > 0.95


def test_uncovered_films_fall_back_to_the_mean(collaborative_setup):
    """MovieLens is silent for the least popular films; that must be explicit."""
    features, factors, links = collaborative_setup
    ratings = pd.Series(np.linspace(1, 5, len(features)))
    model = CollaborativeFold(factors, links).fit(features, ratings)

    covered = model.coverage(features)
    assert covered.sum() == 60
    predictions = model.predict(features)
    assert predictions[~covered] == pytest.approx(np.full((~covered).sum(), ratings.mean()))


class Memoriser:
    """Perfect on rows it trained on, useless on anything else.

    Stands in for a model that overfits its training fold: in-sample it looks
    flawless, out-of-fold it carries no information at all.
    """

    name = "memoriser"

    def fit(self, features, ratings):
        self.table_ = dict(zip(features.index, ratings.astype(float), strict=False))
        self.mean_ = float(ratings.mean())
        return self

    def predict(self, features):
        return np.array([self.table_.get(i, self.mean_) for i in features.index])


def test_blend_weights_are_fitted_out_of_fold(collaborative_setup):
    """The regression test for stacking on in-sample predictions.

    A memorising content leg looks perfect in-sample, so weights fitted that way
    would go entirely to it and the collaborative leg would be discarded. Fitted
    out of fold, the memoriser is revealed as worthless and the weight must move
    to the leg that actually generalises.
    """
    features, factors, links = collaborative_setup
    means = factors.set_index("movieId")["item_mean"]
    ids = pd.to_numeric(features["tmdb_id"], errors="coerce")
    ratings = pd.Series(ids.map(dict(zip(links["tmdbId"], means, strict=False))).fillna(3.0))

    blend = SupportGatedBlend(Memoriser(), CollaborativeFold(factors, links))
    blend.fit(features, ratings)
    assert blend.weight_ > 0.8, "weight should favour the generalising leg"


def test_blend_falls_back_to_content_where_collaborative_is_silent(collaborative_setup):
    features, factors, links = collaborative_setup
    ratings = pd.Series(np.linspace(1, 5, len(features)))
    content = ContentRidge(keyword_components=4, synopsis_components=4)
    blend = SupportGatedBlend(content, CollaborativeFold(factors, links)).fit(features, ratings)

    covered = blend.collaborative.coverage(features)
    blended = blend.predict(features)
    content_only = blend.content.predict(features)
    # Uncovered rows must be exactly the content prediction, not a blend with a
    # constant, which would drag them toward the mean for no reason.
    assert blended[~covered] == pytest.approx(content_only[~covered])


def test_blend_weight_stays_within_bounds(collaborative_setup):
    features, factors, links = collaborative_setup
    rng = np.random.default_rng(3)
    ratings = pd.Series(rng.uniform(1, 5, len(features)))  # pure noise: nothing to learn
    blend = SupportGatedBlend(
        ContentRidge(keyword_components=4, synopsis_components=4),
        CollaborativeFold(factors, links),
    ).fit(features, ratings)
    assert 0.0 <= blend.weight_ <= 1.0


# --------------------------------------------------------------------------
# registry and N-way stacking
# --------------------------------------------------------------------------
def test_registry_builds_every_advertised_content_model():
    from lbrec.models import CONTENT_ONLY, build_model

    for name in CONTENT_ONLY:
        assert build_model(name) is not None


def test_registry_refuses_collaborative_models_without_factors():
    """A clear error beats a model silently predicting the mean for everything."""
    from lbrec.models import build_model

    with pytest.raises(ValueError, match="needs MovieLens item factors"):
        build_model("hybrid")


def test_registry_rejects_an_unknown_name():
    from lbrec.models import build_model

    with pytest.raises(ValueError, match="unknown model"):
        build_model("nonesuch")


def test_stacked_ensemble_weights_are_non_negative_and_named(collaborative_setup):
    from lbrec.models import CrowdScore, StackedEnsemble

    features, factors, links = collaborative_setup
    ratings = pd.Series(np.linspace(1, 5, len(features)))
    ensemble = StackedEnsemble(
        [ContentRidge(keyword_components=4, synopsis_components=4), CrowdScore()]
    ).fit(features, ratings)

    weights = ensemble.describe_weights()
    assert (weights["weight"] >= 0).all()
    assert set(weights["model"]) == {"content_ridge", "crowd_score"}
    assert np.isfinite(ensemble.predict(features)).all()


def test_stacked_ensemble_discards_a_useless_member(collaborative_setup):
    """A memoriser carries no out-of-fold signal, so it should get ~zero weight."""
    from lbrec.models import StackedEnsemble

    features, factors, links = collaborative_setup
    means = factors.set_index("movieId")["item_mean"]
    ids = pd.to_numeric(features["tmdb_id"], errors="coerce")
    ratings = pd.Series(ids.map(dict(zip(links["tmdbId"], means, strict=False))).fillna(3.0))

    ensemble = StackedEnsemble([Memoriser(), CollaborativeFold(factors, links)]).fit(
        features, ratings
    )
    weights = ensemble.describe_weights().set_index("model")["weight"]
    assert weights["collaborative_fold"] > weights["memoriser"]
