"""Tests for the per-user taste diagnostics.

These quantities vary by person, so the tests check the measurement, never a
particular value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbrec.profile import build_profile, popularity_gap, spearman


def make_rated(ratings, votes, crowd=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rating": pd.Series(ratings, dtype="Float64"),
            "vote_count": pd.Series(votes, dtype="Int64"),
            "vote_average": pd.Series(
                crowd if crowd is not None else [5.0] * len(ratings), dtype="Float64"
            ),
        }
    )


def test_spearman_detects_a_monotone_but_nonlinear_relationship():
    x = pd.Series([1.0, 2, 3, 4, 5])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -(x**3)) == pytest.approx(-1.0)


def test_spearman_ignores_rows_with_a_missing_value():
    left = pd.Series([1.0, 2, 3, 4, None])
    right = pd.Series([1.0, 2, 3, 4, 99])
    assert spearman(left, right) == pytest.approx(1.0)


def test_spearman_returns_none_when_undefined():
    assert spearman(pd.Series([1.0, 2]), pd.Series([1.0, 2])) is None  # too few rows
    assert spearman(pd.Series([1.0] * 5), pd.Series(range(5))) is None  # no variance


def test_profile_reports_the_rmse_floor_as_the_rating_sd():
    """Predicting the mean is the easy bar; its RMSE is the standard deviation."""
    rated = make_rated([1.0, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    summary = build_profile(rated, rated).summary.set_index("metric")["value"]
    assert summary["rated films"] == 5
    assert summary["mean rating"] == 3.0
    assert summary["RMSE floor"] == round(float(np.std([1, 2, 3, 4, 5])), 3)


def test_profile_separates_popularity_from_crowd_quality():
    """A user who tracks crowd scores but not fame must show exactly that."""
    rated = make_rated(
        ratings=[1.0, 2, 3, 4, 5],
        votes=[50_000, 10, 30_000, 20, 100],  # unrelated to rating
        crowd=[5.0, 6, 7, 8, 9],  # perfectly ordered with rating
    )
    summary = build_profile(rated, rated).summary.set_index("metric")["value"]
    assert summary["rho(rating, crowd score)"] == pytest.approx(1.0)
    assert abs(summary["rho(rating, log vote_count)"]) < 0.9


def test_deciles_are_cut_against_the_catalogue_not_the_user():
    """Decile 1 must mean the same thing for every user, so bins come from the catalogue."""
    catalogue = pd.DataFrame({"vote_count": pd.Series(range(1000), dtype="Int64")})
    obscure = make_rated([3.0] * 5, [1, 2, 3, 4, 5])
    deciles = build_profile(obscure, catalogue).deciles
    assert deciles["decile"].tolist() == [1]  # all in the lowest catalogue decile
    assert deciles["films"].sum() == 5


def test_tail_share_is_reported_at_several_thresholds():
    rated = make_rated([3.0] * 4, [50, 600, 2000, 20_000])
    tail = build_profile(rated, rated).tail.set_index("under_votes")["films"]
    assert tail[100] == 1
    assert tail[1000] == 2
    assert tail[5000] == 3


def test_profile_survives_a_user_with_no_ratings():
    empty = make_rated([], [])
    result = build_profile(empty, pd.DataFrame({"vote_count": pd.Series([1, 2, 3], dtype="Int64")}))
    assert result.summary.set_index("metric")["value"]["rated films"] == 0
    assert result.deciles.empty


def test_popularity_gap_runs_in_either_direction():
    """A watchlist may be more obscure than the history, or less; both are normal."""
    rated = make_rated([3.0] * 3, [5000, 6000, 7000])
    obscure_wishes = make_rated([None] * 3, [10, 20, 30])
    gap = popularity_gap(rated, obscure_wishes).set_index("set")["median_votes"]
    assert gap["rated"] > gap["watchlist pending"]

    popular_wishes = make_rated([None] * 3, [50_000, 60_000, 70_000])
    gap = popularity_gap(rated, popular_wishes).set_index("set")["median_votes"]
    assert gap["rated"] < gap["watchlist pending"]


def test_popularity_gap_handles_an_empty_watchlist():
    rated = make_rated([3.0], [100])
    gap = popularity_gap(rated, make_rated([], [])).set_index("set")
    # An absent median is NaN once pandas types the column numerically.
    assert pd.isna(gap.loc["watchlist pending", "median_votes"])
