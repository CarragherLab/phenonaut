# Copyright © The University of Edinburgh, 2023.
# Development has been supported by GSK.

import pandas as pd
from phenonaut.data import Dataset
import numpy as np
from phenonaut.phenonaut import Phenonaut
from typing import Optional, Union, Callable
import numpy as np
from scipy.stats import spearmanr
from random import sample
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from warnings import warn


def percent_replicating(
    ds: Union[Dataset, Phenonaut, pd.DataFrame],
    features: Optional[list[str]] = None,
    perturbation_identifier: Optional[str] = None,
    groupby: Optional[list[str]] = None,
    n_iters: int = 1000,
    phenotypic_metric: Union[str, Callable] = "spearman",
    min_number_of_repeats: int = 2,
    return_full_performance_df: bool = False,
    n_jobs: int = -1,
):
    """Calculate percent replicating

    Percent replicating is defined by Way et. al. in:
    Way, Gregory P., et al. "Morphology and gene expression profiling provide complementary
    information for mapping cell state." Cell systems 13.11 (2022): 911-923.
    or on bioRxiv:
    https://www.biorxiv.org/content/10.1101/2021.10.21.465335v2

    For each query compound at a concentration and in a common well (multiple plates), calculate
    pairwise distances and obtain the median.

    Next we compare this against the 95th percentile of a null distribution.
    If the replicates are greater, then it is replicating.

    To create the null distribution, we remove the query compound from the pool of possible
    compounds (at any concentration and in any well). Then we pick the same number of compounds
    as there were repeats of the query compound at a given concentration and in a well. We
    perform this to match the cardinality of repeats across distributions. From the data,
    one profile for each non-matching compound is chosen and then the median recorded when
    compared with a phenoytypic metric (by default Spearman's rank). This creation and scoring
    of the null distribution is performed (by default) 1,000 times.

    The original mean score of the query compound repeats is then compared against the 95th
    percentile of this null distribution. If it is greater, then compound is deemed replicating.

    The percent replicating is then calculated from the number which were replicating versus the
    number which were not.

    As the calculation is demanding, the function makes use of the joblib library for parallel
    calculation of the null distribution.

    Parameters
    ----------
    ds : Union[Dataset, Phenonaut, pd.DataFrame],
        Input data in the form of a Phenonaut dataset, a Phenonaut object, or a pd.DataFrame
        containing profiles of perturbations. If a Phenonaut object is supplied, then the last added
        Dataset will be used.
    features : features:Optional[list[str]]=None,
        Features list which capture the phenotypic responses to perturbations. Only required if
        a pd.DataFrame is supplied in place of the ds argument.  If a Phenonaut object, or Dataset,
        the features will be examined and used as present in the Dataset/Phenonaut object last
        added Dataset. By default None.
    perturbation_identifier : Optional[str]
        Required if a pd.DataFrame is supplied in place of the ds argument. If a Phenonaut object is
        supplied, then the perturbation_identifier is obtained from the perturbation_column property
        of the Dataset. The same happens if a Dataset is supplied. By default, None.
    groupby : Optional[list[str]], optional
        Columns in the dataframe that should be used to group perturbations in addition to the
        perturbation_identifier argument. To replicate percent replicating as defined by Way,
        the column names which capture concentration and well position should be supplied here.
        Way does however define variations of the percent replicating metric, where concentration
        and well position are relaxed. If performing the calculation in this manner, then groupby
        may be None.
    n_iters : int, optional
        Number of times the non-matching compound replicates should be sampled to compose the null
        distribution, by default 1000
    phenotypic_metric : Union[str, Callable], optional
        A callable phenotypic metric may be passed in, or the special case of the string 'spearman'
        which returns the rank score. Other valid and usable text strings may be used which are then
        passed to scikit's pairwise_distances function.  See documentation for usable strings
        defining metrics. By default 'spearman'
    min_number_of_repeats : int, optional
        The minimum number of matched query compounds which must be present to include the treatment
        in the percent replicating summary statistic. By default 2.
    return_full_performance_df : bool
        If True, then a tuple is returned with the percent replicating score in the first position,
        and a pd.DataFrame containing full information on each repeat
    n_jobs : int, optional
        The n_jobs argument is passed to joblib for parallel execution and defines the number of
        cpus to use.  A value of -1 denotes that the system should determine how many jobs to run.
        By default -1.

    Returns
    -------
    Union[float, tuple[float, pd.DataFrame]]
        If return_full_performance_df is False, then only the percent replicating score is returned.
        If True, then a tuple is returned, with percent replicating in the first position, and a
        pd.DataFrame in the second position containing the median repeat scores, as well as median
        null distribution scores in an easy to analyse format.

    """
    if isinstance(ds, Phenonaut):
        ds = ds[-1]

    if isinstance(ds, Dataset):
        df = ds.df
        features = ds.features
        if perturbation_identifier is None:
            if ds.perturbation_column is None:
                raise ValueError(
                    "perturbation_identifier was not supplied, and perturbation_column is not set in the dataset"
                )
            perturbation_identifier = ds.perturbation_column
    elif isinstance(ds, pd.DataFrame):
        df = ds
        if features is None:
            raise ValueError(
                "Must supply features if using a pd.DataFrame to calculate % replicating"
            )
        if perturbation_identifier is None:
            raise ValueError(
                "Must supply perturbation_identifier if using a pd.DataFrame to calculate % replicating"
            )

    if groupby is None:
        groupby = []
    if return_full_performance_df:
        perf_dict = {
            perturbation_identifier: [],
            **{g: [] for g in groupby},
            "median_replicate_score": [],
            "null_95th_percentile": [],
            "is_replicating": [],
            **{f"null_{i+1}": [] for i in range(n_iters)},
        }

    # Private function which will be executed in parallel by joblib
    # Returns a tuple of
    # - median_replicate_score
    # - median_non_replicate_scores
    def __get_cpd_pct_replicating_info(
        df: pd.DataFrame,
        features: list[str],
        g_df: pd.DataFrame,
        g_name: Union[tuple[Union[str, float]], str],
    ):
        # If only grouped by one thing, like a string, then the group 'name' variable is just a string, not a tuple
        # containing the multi index keys.  Therefore we need to properly extract the name
        query_cpd_name = g_name[0] if isinstance(g_name, tuple) else g_name
        cardinality = g_df.shape[0]
        if cardinality < min_number_of_repeats:
            return None

        median_replicate_score = np.median(
            pairwise_distances(g_df[features], metric=phenotypic_metric)[
                np.tril_indices(cardinality, k=-1)
            ]
        )
        median_non_replicate_scores = np.empty(n_iters)
        non_matching_compound_choices = [p for p in perturbation_ids if p != query_cpd_name]

        # If testing a very small pool of compounds with:
        # DMSO x 10
        # Cpd1 x 2
        # Cpd2 x 2
        # Then we can never match the cardinality of the DMSO pool's 10x repeats. Warn about this and then reduce cardinality to 2.
        if cardinality > len(non_matching_compound_choices):
            warn(
                f"Less non-matching compounds than cardinality (repeats of matching treatment), reducing cardinality from {cardinality} to {len(non_matching_compound_choices)}"
            )
            cardinality = len(non_matching_compound_choices)

        for i in range(n_iters):
            sampled_choices = sample(non_matching_compound_choices, k=cardinality)
            null_data = np.vstack(
                [
                    df.query(f"{perturbation_identifier}==@non_matching_perturbation")[features]
                    .sample(1)
                    .values
                    for non_matching_perturbation in sampled_choices
                ]
            )
            median_non_replicate_scores[i] = np.median(
                pairwise_distances(null_data, metric=phenotypic_metric)[
                    np.tril_indices(cardinality, k=-1)
                ]
            )
        return (
            g_name,
            g_df.shape[0],
            cardinality,
            median_replicate_score,
            median_non_replicate_scores,
        )

    samples_replicated = []
    perturbation_ids = df[perturbation_identifier].unique().tolist()

    # spearmanr returns (rank, p-val) tuple, we only want the rank if spearman specified.
    if phenotypic_metric == "spearman":
        phenotypic_metric = lambda x, y: spearmanr(x, y)[0]

    cpd_pct_replicating_info = Parallel(n_jobs=n_jobs)(
        delayed(__get_cpd_pct_replicating_info)(df, features, g_df, g_name)
        for g_name, g_df in df.groupby(
            perturbation_identifier if not groupby else [perturbation_identifier] + groupby
        )
    )

    for t in cpd_pct_replicating_info:
        if t is None:
            continue
        percentile_score = np.percentile(t[4], 95)
        if t[3] > percentile_score:
            samples_replicated.append(1)
        else:
            samples_replicated.append(0)
        if return_full_performance_df:
            perf_dict[perturbation_identifier].append(t[0][0] if isinstance(t[0], tuple) else t[0])
            for gb_i, gb in enumerate(groupby):
                perf_dict[gb].append(t[0][gb_i])
            perf_dict["median_replicate_score"].append(t[3])
            perf_dict["null_95th_percentile"].append(percentile_score)
            perf_dict["is_replicating"].append(samples_replicated[-1])
            for i in range(n_iters):
                perf_dict[f"null_{i+1}"].append(t[4][i])
    percent_replicating_score = (np.sum(samples_replicated) / len(samples_replicated)) * 100
    if return_full_performance_df:
        return (percent_replicating_score, pd.DataFrame(perf_dict))
    else:
        return percent_replicating_score
