from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cityblock, cosine, euclidean
from scipy.stats import rankdata, spearmanr


def calc_zhang_scores(anchor: np.array, queries: np.array) -> Union[float, np.ndarray]:
    """Calculate Zhang scores between two np.ndarrays

    Implementation of the Zhang method for comparing L1000/CMAP signatures.
    Zhang, Shu-Dong, and Timothy W. Gant. "A simple and robust method for
    connecting small-molecule drugs using gene-expression signatures." BMC
    bioinformatics 9.1 (2008): 1-10.
    Implemented by Steven Shave, following above paper as a reference
    https://doi.org/10.1186/1471-2105-9-258

    Parameters
    ----------
    anchor : np.array
        Anchor profiles/features. Can be a MxN matrix, allowing M sequences to
        be queried against queries (using N features).
    queries : np.array
        Candidate profiles/features. Can be a MxN matrix, allowing M candidate
        sequences to be evaluated against anchor sequences.

    Returns
    -------
    Union[float, np.ndarray]
        If anchor and candidate array ndims are both 1, then a single float
        representing the Zhang score is returned. If one input array has ndims
        of  2 (and the other has ndims of 1), then a 1-D np.ndarray is
        returned. If both inputs are 2-D, then a 2D MxN array is returned, where
        M is the


    """
    if queries is None:
        queries = anchor
    if anchor.ndim != 1:
        if anchor.ndim == 2:
            multi_anchor_results = np.full((anchor.shape[0], queries.shape[0]), np.nan)
            for i in range(anchor.shape[0]):
                multi_anchor_results[i, :] = calc_zhang_scores(anchor[i], queries)
            return multi_anchor_results
        else:
            raise ValueError(
                f"Anchor should be a 1D array, it had shape: {anchor.shape}"
            )
    anchor_profile = rankdata(np.abs(anchor), axis=-1) * np.sign(anchor)
    if queries.ndim == 1:
        queries.reshape(1, -1)
    if queries.shape[-1] != anchor.shape[-1]:
        raise ValueError(
            f"Different number of features found in anchor ({anchor.shape[-1]}) and queries ({queries.shape[-1]})"
        )
    query_profiles = rankdata(np.abs(queries), axis=-1) * np.sign(queries)
    return np.sum(anchor_profile * query_profiles, axis=-1) / np.sum(
        anchor_profile**2, axis=-1
    )


def calc_zhang_scores_all_v_all(anchor: np.array):
    anchor = np.array(anchor)
    if anchor.ndim == 2:
        multi_anchor_results = np.full((anchor.shape[0], anchor.shape[0]), np.nan)
        query_profiles = rankdata(np.abs(anchor), axis=-1) * np.sign(anchor)
        for i in range(anchor.shape[0]):
            anchor_profile = rankdata(np.abs(anchor[i]), axis=-1) * np.sign(anchor[i])
            multi_anchor_results[i, :] = np.sum(
                anchor_profile * query_profiles, axis=-1
            ) / np.sum(anchor_profile**2, axis=-1)
        return multi_anchor_results
    else:
        raise ValueError(f"Anchor should be a 2D array, it had shape: {anchor.shape}")


def calc_spearmansrank_scores(
    anchor: np.array, queries: np.array
) -> Union[float, np.ndarray]:
    # Standardise inputs to np arrays
    anchor = np.array(anchor)
    if queries is None:
        queries = np.array(anchor)
    else:
        queries = np.array(queries)

    if anchor.ndim == 1:
        if queries.ndim == 1:
            return spearmanr(anchor, queries).correlation
        else:
            return np.array([spearmanr(anchor, q).correlation for q in queries])
    else:
        if anchor.ndim != 2:
            raise ValueError("Anchor should be 1D or 2D, it was {anchor.ndim}D")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        results_array = np.empty((anchor.shape[0], queries.shape[0]))

        for i in range(anchor.shape[0]):
            results_array[i] = np.array(
                [spearmanr(anchor[i], q).correlation for q in queries]
            )
        return results_array


def calc_connectivity_scores(
    anchor: np.array, queries: np.array
) -> Union[float, np.ndarray]:
    # Standardise inputs to np arrays
    anchor = np.array(anchor)
    if queries is None:
        queries = np.array(anchor)
    else:
        queries = np.array(queries)

    if anchor.ndim == 1:
        if queries.ndim == 1:
            return np.sum(np.sign(anchor) == np.sign(queries)) / len(anchor)
        else:
            return np.array(
                [np.sum(np.sign(anchor) == np.sign(q)) / len(anchor) for q in queries]
            )
    else:
        if anchor.ndim != 2:
            raise ValueError("Anchor should be 1D or 2D, it was {anchor.ndim}D")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        results_array = np.empty((anchor.shape[0], queries.shape[0]))

        for i in range(anchor.shape[0]):
            results_array[i] = np.array(
                [
                    np.sum(np.sign(anchor[i]) == np.sign(q)) / len(anchor[i])
                    for q in queries
                ]
            )
        return results_array


class PhenotypicMetric:
    """Metrics evaluate one profile/feature vector against another

    SciPy and other libraries traditionally supply distance metrics, like
    Manhattan, Euclidean etc. These are typically unbound in their max value,
    but not always, for example; cosine distance with a maximum dissimilarity of
    1.  Scientific literature is also full of similarity metrics, where a high
    value indicates most similarity - the opposite of a similarity metric. This
    dataclass coerces metrics into a standard form, with .similarity and
    .distance functions to turn any metric into a similarity or distance metric.

    This allows the definition of something like the Zhang similarity metric,
    which ranges from -1 to 1, indicating most dissimilarity and most similarity
    respectively. Calling the metric defined by this Zhang function will return
    the traditional Zhang metric value - ranging from -1 to 1.

    The methods similarity and distance will also be added

    Calling distance will return a value between 0 and 1, with 0 being most
    similar and 1 being most dissimilar.

    Calling similarity will return a value between 0 and 1, with 1 being the
    most similar and 0 being the most different.

    """

    def __init__(
        self,
        name: str,
        method: Union[Callable, str],
        range: Tuple[float, float],
        higher_is_better: bool = True,
    ):
        self.name = name
        self.func = method
        self.range = range
        self.higher_is_better = higher_is_better
        if isinstance(self.func, str):
            self.is_magic_string = True
        else:
            self.is_magic_string = False
        if not any(np.isinf(self.range)):
            self.scalable = True
        else:
            self.scalable = False

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __call__(self, anchor, query):
        anchor = np.array(anchor)
        query = np.array(query)
        if anchor.ndim != 1:
            raise ValueError(
                f"Expected anchor to have 1 dimension, it had {anchor.ndim}"
            )
        if query.ndim == 2:
            return np.array([self.__call__(anchor, row) for row in query])
        return self.func(anchor, query)

    def scale(self, score):
        return (score - self.range[0]) / (self.range[1] - self.range[0])

    def distance(self, anchor, query):
        score = self.__call__(anchor, query)

        if self.scalable:
            score = self.scale(score)
            if self.higher_is_better:
                return 1.0 - score
            return score
        else:
            if self.higher_is_better:
                return 1.0 / score
            return score

    def similarity(self, anchor, query):
        score = self.__call__(anchor, query)
        if self.scalable:
            score = self.scale(score)
            if self.higher_is_better:
                return score
            else:
                return 1 - score
        else:
            if self.higher_is_better:
                return score
            else:
                if isinstance(score, float):
                    if score == 0:
                        return 1
                    return 1.0 / score
                return 1.0 / np.clip(score, np.finfo(float).eps, None)


non_ds_phenotypic_metrics = {
    "Connectivity": PhenotypicMetric("Connectivity", calc_connectivity_scores, (-1, 1)),
    "Rank": PhenotypicMetric("Rank", calc_spearmansrank_scores, (-1, 1)),
    "Zhang": PhenotypicMetric("Zhang", calc_zhang_scores, (-1, 1)),
    "Euclidean": PhenotypicMetric(
        "Euclidean", euclidean, (0, np.inf), higher_is_better=False
    ),
    "Manhattan": PhenotypicMetric(
        "Manhattan", cityblock, (0, np.inf), higher_is_better=False
    ),
    "Cosine": PhenotypicMetric("Cosine", cosine, (0, 2), higher_is_better=False),
}

# Metrics with magic values - magic values referrs to the fast methods which  scipy's pdist/cidist use
metrics_with_magic_values = {
    "Rank": PhenotypicMetric("Rank", "spearman", (-1, 1)),
    "Euclidean": PhenotypicMetric(
        "Euclidean", "euclidean", (0, np.inf), higher_is_better=False
    ),
    "Manhattan": PhenotypicMetric(
        "Manhattan", "cityblock", (0, np.inf), higher_is_better=False
    ),
    "Cosine": PhenotypicMetric("Cosine", "cosine", (0, 2), higher_is_better=False),
    "Connectivity": PhenotypicMetric("Connectivity", calc_connectivity_scores, (-1, 1)),
    "Zhang": PhenotypicMetric("Zhang", calc_zhang_scores, (-1, 1)),
}
