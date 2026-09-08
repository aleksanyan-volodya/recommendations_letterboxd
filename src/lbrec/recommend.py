"""Score the catalogue and produce a ranked list.

The first stage that actually recommends something. Everything before this could
only score films already in the export, which is why so much of the project has
been optimising a proxy without ever seeing output.

Two honesty constraints shape this module:

**Train and score on the same features.** The catalogue carries no keywords,
cast or crew, so the retrieval model is trained on ``CATALOGUE_BLOCKS`` -- the
intersection of what the export and the catalogue both have. Scoring 285k films
with a model whose keyword coefficients were learned on films that had keywords
would be a train/inference mismatch presented as a recommendation.

**Ranking mode is explicit.** ``plain`` ranks by predicted rating and will
inevitably favour the well-known, because popularity correlates with everything
the model can see. ``stratified`` takes the best from each popularity decile,
which is the literal reading of "fair relative to quality, not obscurity".
``calibrated`` reranks to hit a target share of long-tail films. The mode is a
parameter rather than a default, because which one is right is a judgement about
what the list is *for*, not something the data settles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lbrec.catalogue import BAND_LABELS, TAIL_BANDS, popularity_band
from lbrec.features import CATALOGUE_BLOCKS, build_film_features

#: How the ranked list is assembled from raw predictions.
RANK_PLAIN = "plain"
RANK_STRATIFIED = "stratified"
RANK_CALIBRATED = "calibrated"
RANK_MODES = (RANK_PLAIN, RANK_STRATIFIED, RANK_CALIBRATED)

DISPLAY_COLUMNS = [
    "rank",
    "title",
    "year",
    "score",
    "vote_count",
    "band",
    "vote_average",
    "original_language",
    "tmdb_id",
]


@dataclass(frozen=True)
class Recommendations:
    films: pd.DataFrame
    model_name: str
    n_scored: int
    n_excluded: int


def _stratified(scored: pd.DataFrame, k: int, bins: int) -> pd.DataFrame:
    """Best few from every popularity decile.

    Guarantees the obscure end of the catalogue is represented at all, instead
    of being competed out by films the model can simply say more about.
    """
    per_band = max(1, k // len(BAND_LABELS))
    picked = (
        scored.sort_values("score", ascending=False)
        .groupby("band", observed=True, sort=False)
        .head(per_band)
    )
    # Any shortfall (a decile with too few candidates) is filled on merit.
    if len(picked) < k:
        remainder = scored[~scored.index.isin(picked.index)].nlargest(k - len(picked), "score")
        picked = pd.concat([picked, remainder])
    return picked.sort_values("score", ascending=False).head(k)


def _calibrated(scored: pd.DataFrame, k: int, tail_share: float, tail_deciles: int) -> pd.DataFrame:
    """Fill a target proportion of the list from the long tail, rest on merit.

    Follows Steck's calibrated recommendation: state the distribution you want
    and rerank to meet it, rather than hoping a penalty term produces it. The
    target is a product decision made explicit.
    """
    is_tail = scored["band"].isin(TAIL_BANDS)
    n_tail = min(int(round(k * tail_share)), int(is_tail.sum()))
    tail = scored[is_tail].nlargest(n_tail, "score")
    head = scored[~is_tail].nlargest(k - len(tail), "score")
    return pd.concat([tail, head]).sort_values("score", ascending=False)


def recommend(
    model,
    rated_features: pd.DataFrame,
    ratings: pd.Series,
    catalogue: pd.DataFrame,
    *,
    k: int = 20,
    mode: str = RANK_PLAIN,
    tail_share: float = 0.4,
    tail_deciles: int = 3,
    bins: int = 10,
) -> Recommendations:
    """Fit on the user's ratings, score every catalogue film, return the top k.

    ``catalogue`` must already exclude films the user has seen -- see
    ``catalogue.exclude_seen``, which uses the union of watched, rated and
    diarised rather than ``watched.csv`` alone.
    """
    if mode not in RANK_MODES:
        raise ValueError(f"unknown ranking mode {mode!r}; expected one of {RANK_MODES}")
    if catalogue.empty:
        return Recommendations(pd.DataFrame(columns=DISPLAY_COLUMNS), model.name, 0, 0)

    model.fit(rated_features, ratings)
    candidate_features = build_film_features(catalogue)
    scores = np.asarray(model.predict(candidate_features), dtype=float)

    scored = catalogue.copy()
    scored["score"] = scores
    scored["band"] = popularity_band(scored["vote_count"])
    scored = scored[np.isfinite(scored["score"])]

    if mode == RANK_STRATIFIED:
        chosen = _stratified(scored, k, bins)
    elif mode == RANK_CALIBRATED:
        chosen = _calibrated(scored, k, tail_share, tail_deciles)
    else:
        chosen = scored.nlargest(k, "score")

    chosen = chosen.reset_index(drop=True)
    chosen.insert(0, "rank", np.arange(1, len(chosen) + 1))
    chosen["score"] = chosen["score"].round(3)
    present = [column for column in DISPLAY_COLUMNS if column in chosen.columns]
    return Recommendations(chosen[present], model.name, len(scored), 0)


def retrieval_blocks() -> tuple[str, ...]:
    """Feature blocks a catalogue-scoring model may use."""
    return CATALOGUE_BLOCKS


def summarise_popularity(films: pd.DataFrame) -> pd.DataFrame:
    """Where a recommended list sits on the popularity axis.

    Printed alongside every list, because a list that looks adventurous can
    still be entirely drawn from the top two deciles, and the titles alone will
    not reveal that.
    """
    if films.empty or "band" not in films:
        return pd.DataFrame(columns=["band", "films"])
    return (
        films.groupby("band", observed=True)
        .size()
        .reset_index(name="films")
        .sort_values("band", ignore_index=True)
    )
