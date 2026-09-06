"""Diagnostics describing one user's taste against catalogue popularity.

These are per-user quantities, not project constants. One person's watchlist may
be long and full of obscurities while another's is short and entirely
mainstream, and the two need different handling. So the numbers are computed
from whichever export is loaded rather than assumed.

Three of them decide how the modelling should go:

``rho(rating, crowd score)``
    The crowd-quality baseline. If a model cannot beat "predict what everyone
    else thought", it has learned nothing personal. This is the bar, not the
    global-mean RMSE, which is far easier to clear.
``rho(rating, log popularity)``
    How much of this user's taste fame already explains. Near zero means a
    popularity-biased model is not merely suboptimal for them, it is wrong.
``tail share``
    Whether there is a long tail in their history at all. If everything they
    have rated sits in the top deciles, there is nothing to surface and the
    debiasing work has no purchase for that user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Vote-count thresholds reported as "how obscure does this library get".
TAIL_THRESHOLDS = (100, 500, 1000, 5000)


def spearman(left: pd.Series, right: pd.Series) -> float | None:
    """Rank correlation over rows where both values are present.

    Implemented directly to keep scipy out of the dependency set; ties get
    average ranks, which is what ``rankdata`` would do.
    """
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return None
    a = frame["left"].astype(float).rank().to_numpy()
    b = frame["right"].astype(float).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class Profile:
    summary: pd.DataFrame
    deciles: pd.DataFrame
    tail: pd.DataFrame


def _decile_edges(votes: pd.Series, bins: int) -> np.ndarray:
    return np.unique(np.quantile(votes.dropna().astype(float), np.linspace(0, 1, bins + 1)))


def build_profile(rated: pd.DataFrame, catalogue: pd.DataFrame, *, bins: int = 10) -> Profile:
    """Describe a user's ratings against the popularity of the films they cover.

    ``rated`` needs ``rating`` and ``vote_count`` (plus ``vote_average`` for the
    crowd baseline); ``catalogue`` supplies the popularity bins so the user's
    films are placed against the whole catalogue rather than against themselves.
    """
    rows: list[dict] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    ratings = rated["rating"].dropna().astype(float)
    add("rated films", len(ratings))
    if len(ratings):
        add("mean rating", round(float(ratings.mean()), 3))
        add("rating sd", round(float(ratings.std(ddof=0)), 3))
        add(
            "RMSE floor",
            round(float(ratings.std(ddof=0)), 3),
            "predicting the mean; the easy bar",
        )

    log_votes = np.log1p(rated["vote_count"].astype("Float64").astype(float))
    rho_popularity = spearman(rated["rating"], pd.Series(log_votes, index=rated.index))
    rho_crowd = spearman(rated["rating"], rated.get("vote_average"))
    add(
        "rho(rating, log vote_count)",
        None if rho_popularity is None else round(rho_popularity, 3),
        "how much fame already explains this taste",
    )
    add(
        "rho(rating, crowd score)",
        None if rho_crowd is None else round(rho_crowd, 3),
        "THE baseline to beat",
    )

    # Popularity deciles, cut against the whole catalogue rather than the
    # user's own films, so "decile 1" means the same thing for everyone.
    edges = _decile_edges(catalogue["vote_count"], bins)
    if len(edges) > 2 and len(rated):
        labels = range(1, len(edges))
        placed = pd.cut(
            rated["vote_count"].astype("Float64").astype(float),
            bins=edges,
            labels=labels,
            include_lowest=True,
        )
        deciles = (
            pd.DataFrame({"decile": placed, "rating": rated["rating"]})
            .groupby("decile", observed=True)
            .agg(
                films=("rating", "size"), rated=("rating", "count"), mean_rating=("rating", "mean")
            )
            .reset_index()
        )
        deciles["share"] = (deciles["films"] / len(rated)).map("{:.1%}".format)
        deciles["mean_rating"] = deciles["mean_rating"].round(2)
    else:
        deciles = pd.DataFrame(columns=["decile", "films", "rated", "mean_rating", "share"])

    tail_rows = []
    for threshold in TAIL_THRESHOLDS:
        count = int((rated["vote_count"].astype("Float64") < threshold).sum())
        tail_rows.append(
            {
                "under_votes": threshold,
                "films": count,
                "share": f"{count / len(rated):.1%}" if len(rated) else "",
            }
        )
    tail = pd.DataFrame(tail_rows)

    return Profile(summary=pd.DataFrame(rows), deciles=deciles, tail=tail)


def popularity_gap(rated: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
    """Median popularity of what a user has watched versus what they want next.

    Watchlists are usually assumed to be availability-biased toward well-known
    titles. That is not universal (some users' watchlists are markedly more
    obscure than their history...) and which way it runs decides whether the
    watchlist is safe to use as a held-out positive set for that user.
    """
    rows = []
    for label, frame in (("rated", rated), ("watchlist pending", pending)):
        votes = frame["vote_count"].astype("Float64").dropna()
        rows.append(
            {
                "set": label,
                "films": len(frame),
                "median_votes": int(votes.median()) if len(votes) else None,
                "p10_votes": int(votes.quantile(0.10)) if len(votes) else None,
                "p90_votes": int(votes.quantile(0.90)) if len(votes) else None,
            }
        )
    return pd.DataFrame(rows)
