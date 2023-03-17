import numpy as np
import pandas as pd
from phenonaut.data import Dataset
from phenonaut import Phenonaut
from typing import Optional, Union, Callable
import numpy as np
from random import sample, choice
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from progressbar import progressbar
from pathlib import Path
from collections import namedtuple
import json
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from timeit import default_timer as timer

_ReplicateData = namedtuple(
    "_ReplicateData",
    ["index", "cardinality", "matching_median_dist", "null_nth_percentile_or_null_distribution"],
)


def _check_performance_path_and_gen_if_needed(filename, args_dict: dict, prefix:str) -> Path:
    """Private function, check target performance_path

    It could be a directory, in which case, generate a suitable filename
    from args_dict.  If a file, then just return it.  If a bool, then
    generate a suitable filename.

    Parameters
    ----------
    filename : Union[bool, str, Path]
        Input filename. If True or a directory, generate a suitable filename.
        If a file, then return it.
    args_dict : dict
        Arguments dictionary capturing what percent_replicating was called with.
    prefix : str
        File prefix if auto-generated. Typically, percent_replicating will use
        this function and pass "pr". In the case of percent compact, it will
        pass "pc".

    Returns
    -------
    Path
        Original, or generated performance datafile path.
    """

    def _get_fn(a: dict) -> str:
        fn = f"{prefix}_{a['similarity_metric_name']}__"
        if a["additional_captured_params"] is not None:
            for k, v in a["additional_captured_params"].items():
                fn += f"{k.replace('_','')}_{v.replace('_','')}__"
        if len(a["replicate_criteria"]) > 0:
            fn += "repcrit_"
            for c in a["replicate_criteria"]:
                fn += f"{c.replace('_','')}_"
            fn += "_"

        if len(a["replicate_criteria_not"]) > 0:
            fn += "repcritnot_"
            for c in a["replicate_criteria_not"]:
                fn += f"{c.replace('_','')}_"
            fn += "_"

        if len(a["null_criteria"]) > 0:
            fn += "nullcrit_"
            for c in a["null_criteria"]:
                fn += f"{c.replace('_','')}_"
            fn += "_"
        if len(a["null_criteria_not"]) > 0:
            fn += "nullcritnot_"
            for c in a["null_criteria_not"]:
                fn += f"{c.replace('_','')}_"
            fn += "_"
        if fn[-2:] == "__":
            fn = fn[:-2]
        fn = fn + ".csv"
        return fn

    if isinstance(filename, bool):
        filename = _get_fn(args_dict)
    elif isinstance(filename, str):
        filename = Path(filename)

    if filename.is_dir():
        filename = filename / _get_fn(args_dict)
    if not filename.parent.exists():
        filename.mkdir(parents=True)

    while filename.exists():
        filename = Path(f"{str(filename)[:-4]}_1.csv")
    return filename


def _inspect_replicating(
    df: pd.DataFrame,
    features: list[str],
    g_indices: list[int],
    replicate_criteria: list[str],
    replicate_criteria_not: list[str],
    null_criteria: list[str],
    null_criteria_not: list[str],
    replicate_query: Optional[str] = None,
    null_query_or_df: Optional[Union[str, pd.DataFrame]] = None,
    similarity_metric: Union[str, Callable] = "spearman",
    n_iters: int = 1000,
    min_cardinality: int = 2,
    max_cardinality: int = 50,
    return_full_profiles: bool = False,
    percentile_cutoff: int = 95,
):
    """Private function to obtain replicating info for a single treatment

    Parameters
    ----------
    df : pd.DataFrame
        Input pd.DataFrame
    features : list[str]
        Feature columns within the DataFrame
    g_indices : list[int]
        Operate on the treatment found at these df.loc compatible indices.
    replicate_criteria : list[str]
        Criteria used to define matching
    null_criteria : list[str]
        Criteria which the null distribution must match
    replicate_criteria_not : list[str]
        Criteria upon which matched replicates must not match
    replicate_query: Optional[str]=None
        Optional pandas query to apply in selection of the matching replicates, this maybe something
        like ensuring concentration is above a threshold, or that they are taken from certain timepoints.
        Please note, if information in rows is never going to be included in matching or null distributions,
        then it is more efficient to prefilter the dataframe before running percent_replicating on it.
        This parameter should not be used to restrict the compounds on which percent replicating is run,
        as this is inefficient. Instead, the restrict_evaluation_query should be used.
    null_query_or_df : Optional[str, pd.DataFrame]=None
        Optional pandas query to apply in selection of the non-matching replicates comprising the null
        distribution. This can be things like ensuring only a certain plates or cell lines are used in
        construction of the distribution. Alternatively, a pd.DataFrame may be supplied here, from which
        the non-matching compounds are drawn for creation of the null distribution. Please note - if
        supplying a query, then if information in rows is never going to be included in matching or
        null distributions, then it is more efficient to prefilter the dataframe before running
        percent_replicating on it. By default, None.
    similarity_metric : Union[str, Callable]
        Callable metric, or string which is passed to pdist. This should be a
        similarity metric (higher is better).
        Note, a special case exists, whereby 'spearman' may be supplied here
        if so, then a much faster Numpy method np.corrcoef is used over the standard
        approach of calling pdist.  Defaults to 'spearman'
    n_iters : int, optional
        Number of non-matching compound profiles to measure, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of matched compounds to be included in results,
        if less than this threshold, then it is deemed non-replicating, by default 2.
    max_cardinality : int, optional
        If a dataset has thousands of matched repeats, then little is gained in
        finding pairwise all-to-all distances of non-matching compounds, this
        argument allows setting an upper bound cutoff after which, the repeats
        are shuffled and max_cardinality samples drawn to create a synthetic set of
        max_cardinality repeats. This is very useful when using a demanding similarity
        method as it cuts the evaluations dramatically. By default 50.
    return_full_profiles : bool, optional
        Instead of a percentile cutoff value, return the full null distribution, by default False.
    percentile_cutoff : int
        Percentile of the null distribution over which the matching replicates must score to be
        considered replicating. Should range from 0 to 100. By default 95.

    Returns
    -------
    _ReplicateData
        Named tuple containing information on the replicate performance of the tuple.
        If return_full_profiles is True, then named tuples contain the full null
        distribution, if False, then just np.nan in this position.
        _ReplicateData has the following fields:
        - index (multiindex identifier - eg a tuple of compoundID and dose
        - cardinality - number of repeats included in calculation
        - matching_median_dist - median all to all score of the replicates
        - null_nth_percentile_or_null_distribution - Nth percentile cutoff of
            the null distribution or if requested (return_full_profiles = True),
            the full null distribution.
    """

    matching_multiindex, matching_indices = g_indices
    replicate_frame = df.iloc[matching_indices]

    # Early exit if required, before further checks
    if len(matching_indices) < 2 or len(matching_indices) < min_cardinality:
        return _ReplicateData(matching_multiindex, len(matching_indices), np.nan, np.nan)

    if replicate_query is not None:
        replicate_frame = replicate_frame.query(replicate_query)

    # If replicate_criteria_not contains fields not to match on, then we must regenerate 'matching_indices'
    if len(replicate_criteria_not) > 0:
        matching_indices = [
            choice(nmi)
            for nmi in [
                i_idx for i_idx in replicate_frame.groupby(replicate_criteria_not).indices.values()
            ]
        ]
        replicate_frame = replicate_frame.iloc[matching_indices]

    cardinality = replicate_frame.shape[0]
    # If only 1 compound, then it is not replicating
    if cardinality == 1:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)
    # If less than required cardinality, then return as non-replicating
    if cardinality < min_cardinality:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)

    # If matching cardinality is > max_cardinality, then resample to max_cardinality
    if cardinality > max_cardinality:
        replicate_frame = replicate_frame.sample(n=max_cardinality)
        cardinality = max_cardinality

    if cardinality < 2:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)

    # Compose query for selecting null distribution. This can be as simple as
    # not equal to the same compound/pert_id as the matched compound, or include
    # additional terms, like being in the same well as the matched compound. Or
    # in the case of null_criteria_not, ensuring things do not match.
    null_criteria_query_list = [f"{replicate_criteria[0]}!=@matching_multiindex[0]"]
    if len(null_criteria) > 0:
        tmp_matched_cpd_row = replicate_frame.iloc[0]
        for gbn in null_criteria:
            tmp_val = tmp_matched_cpd_row[gbn]
            if isinstance(tmp_val, (float, int)):
                null_criteria_query_list.append(f"{gbn}=={tmp_val}")
            else:
                null_criteria_query_list.append(f"{gbn}=='{tmp_val}'")
    # Generate non-matching queries
    if len(null_criteria_not) > 0:
        tmp_matched_cpd_row = replicate_frame.iloc[0]
        for gbn in null_criteria_not:
            tmp_val = tmp_matched_cpd_row[gbn]
            if isinstance(tmp_val, (float, int)):
                null_criteria_query_list.append(f"{gbn}!={tmp_val}")
            else:
                null_criteria_query_list.append(f"{gbn}!='{tmp_val}'")

    if isinstance(null_query_or_df, pd.DataFrame):
        null_df = null_query_or_df.query(" and ".join(null_criteria_query_list))
    else:
        null_df = df.query(" and ".join(null_criteria_query_list))
        if isinstance(null_query_or_df, str):
            null_df = null_df.query(null_query_or_df)

    # generate a dictionary (pert_name_to_idx_dict), allowing mapping from pert_name
    # to indexes (0 -indexed np array)
    null_pert_names = null_df[replicate_criteria[0]]
    pert_name_to_idx_dict = {pname: [] for pname in null_pert_names.unique()}
    for i, pert_name in enumerate(null_pert_names):
        pert_name_to_idx_dict[pert_name].append(i)
    if len(pert_name_to_idx_dict) < 2 or len(pert_name_to_idx_dict) < min_cardinality:
        return _ReplicateData(matching_multiindex, len(matching_indices), np.nan, np.nan)

    if cardinality > len(pert_name_to_idx_dict):
        print(
            f"Number of unique non-matching compounds ({len(pert_name_to_idx_dict)}) was less than the number of repeats ({cardinality}) for {matching_multiindex}, running with a reduced cardinality of {len(pert_name_to_idx_dict)}"
        )
        cardinality = len(pert_name_to_idx_dict)
        replicate_frame = replicate_frame.sample(n=cardinality)

    # Calculate matched repeat score
    if similarity_metric == "spearman":
        matching_median_dist = np.median(
            np.corrcoef(np.argsort(np.argsort(replicate_frame[features].values)))[
                np.triu_indices(cardinality, k=1)
            ]
        )
    else:
        matching_median_dist = np.median(
            pdist(replicate_frame[features].values, metric=similarity_metric)
        )

    # If c is cardinality, choose c non-matching compounds. For each c, choose a
    # random repeat index r. Compose list of length n_iters x c.
    indices = [
        choice(pert_name_to_idx_dict[cpd])
        for _ in range(n_iters)
        for cpd in sample(list(pert_name_to_idx_dict.keys()), cardinality)
    ]

    if len(indices) < 2:
        return _ReplicateData(matching_multiindex, len(matching_indices), np.nan, np.nan)

    # Extract the profiles then reahape to a 3D array n_iters x cardinality x features
    profiles = null_df[features].values[indices].reshape(n_iters, cardinality, len(features))

    # Calculate the null distribution
    if similarity_metric == "spearman":
        null_distribution = np.array(
            [
                np.median(
                    np.corrcoef(np.argsort(np.argsort(profiles[p])))[
                        np.triu_indices(cardinality, k=1)
                    ]
                )
                for p in range(n_iters)
            ]
        )
    else:
        null_distribution = np.array(
            [np.median(pdist(profiles[p], metric=similarity_metric)) for p in range(n_iters)]
        )

    return _ReplicateData(
        matching_multiindex,
        cardinality,
        matching_median_dist,
        null_distribution
        if return_full_profiles
        else np.percentile(null_distribution, float(percentile_cutoff)),
    )


def _inspect_compact(
    df: pd.DataFrame,
    features: list[str],
    g_indices: list[int],
    replicate_criteria: list[str],
    replicate_criteria_not: list[str],
    null_criteria: list[str],
    null_criteria_not: list[str],
    replicate_query: Optional[str] = None,
    null_query_or_df: Optional[Union[str, pd.DataFrame]] = None,
    similarity_metric: Union[str, Callable] = "spearman",
    n_iters: int = 1000,
    min_cardinality: int = 2,
    max_cardinality: int = 50,
    return_full_profiles: bool = False,
    percentile_cutoff: int = 95,
):
    """Private function to obtain compactness info for a single treatment

    Parameters
    ----------
    df : pd.DataFrame
        Input pd.DataFrame
    features : list[str]
        Feature columns within the DataFrame
    g_indices : list[int]
        Operate on the treatment found at these df.loc compatible indices.
    replicate_criteria : list[str]
        Criteria used to define matching
    null_criteria : list[str]
        Criteria which the null distribution must match
    replicate_criteria_not : list[str]
        Criteria upon which matched replicates must not match
    replicate_query: Optional[str]=None
        Optional pandas query to apply in selection of the matching replicates, this maybe something
        like ensuring concentration is above a threshold, or that they are taken from certain timepoints.
        Please note, if information in rows is never going to be included in matching or null distributions,
        then it is more efficient to prefilter the dataframe before running percent_replicating on it.
        This parameter should not be used to restrict the compounds on which percent replicating is run,
        as this is inefficient. Instead, the restrict_evaluation_query should be used.
    null_query_or_df : Optional[str, pd.DataFrame]=None
        Optional pandas query to apply in selection of the non-matching replicates comprising the null
        distribution. This can be things like ensuring only a certain plates or cell lines are used in
        construction of the distribution. Alternatively, a pd.DataFrame may be supplied here, from which
        the non-matching compounds are drawn for creation of the null distribution. Please note - if
        supplying a query, then if information in rows is never going to be included in matching or
        null distributions, then it is more efficient to prefilter the dataframe before running
        percent_replicating on it. By default, None.
    similarity_metric : Union[str, Callable]
        Callable metric, or string which is passed to pdist. This should be a
        similarity metric (higher is better).
        Note, a special case exists, whereby 'spearman' may be supplied here
        if so, then a much faster Numpy method np.corrcoef is used over the standard
        approach of calling pdist.  Defaults to 'spearman'
    n_iters : int, optional
        Number of non-matching compound profiles to measure, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of matched compounds to be included in results,
        if less than this threshold, then it is deemed non-replicating, by default 2.
    max_cardinality : int, optional
        If a dataset has thousands of matched repeats, then little is gained in
        finding pairwise all-to-all distances of non-matching compounds, this
        argument allows setting an upper bound cutoff after which, the repeats
        are shuffled and max_cardinality samples drawn to create a synthetic set of
        max_cardinality repeats. This is very useful when using a demanding similarity
        method as it cuts the evaluations dramatically. By default 50.
    return_full_profiles : bool, optional
        Instead of a percentile cutoff value, return the full null distribution, by default False.
    percentile_cutoff : int
        Percentile of the null distribution over which the matching replicates must score to be
        considered replicating. Should range from 0 to 100. By default 95.

    Returns
    -------
    _ReplicateData
        Named tuple containing information on the replicate performance of the tuple.
        If return_full_profiles is True, then named tuples contain the full null
        distribution, if False, then just np.nan in this position.
        _ReplicateData has the following fields:
        - index (multiindex identifier - eg a tuple of compoundID and dose
        - cardinality - number of repeats included in calculation
        - matching_median_dist - median all to all score of the replicates
        - null_nth_percentile_or_null_distribution - Nth percentile cutoff of
            the null distribution or if requested (return_full_profiles = True),
            the full null distribution.
    """

    matching_multiindex, matching_indices = g_indices
    replicate_frame = df.iloc[matching_indices]

    # Early exit if required, before further checks
    if len(matching_indices) < 2 or len(matching_indices) < min_cardinality:
        return _ReplicateData(matching_multiindex, len(matching_indices), np.nan, np.nan)

    if replicate_query is not None:
        replicate_frame = replicate_frame.query(replicate_query)

    # If replicate_criteria_not contains fields not to match on, then we must regenerate 'matching_indices'
    if len(replicate_criteria_not) > 0:
        matching_indices = [
            choice(nmi)
            for nmi in [
                i_idx for i_idx in replicate_frame.groupby(replicate_criteria_not).indices.values()
            ]
        ]
        replicate_frame = replicate_frame.iloc[matching_indices]

    cardinality = replicate_frame.shape[0]
    # If only 1 compound, then it is not replicating
    if cardinality == 1:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)
    # If less than required cardinality, then return as non-replicating
    if cardinality < min_cardinality:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)

    # If matching cardinality is > max_cardinality, then resample to max_cardinality
    if cardinality > max_cardinality:
        replicate_frame = replicate_frame.sample(n=max_cardinality)
        cardinality = max_cardinality

    if cardinality < 2:
        return _ReplicateData(matching_multiindex, cardinality, np.nan, np.nan)

    # Compose query for selecting null distribution. This can be as simple as
    # not equal to the same compound/pert_id as the matched compound, or include
    # additional terms, like being in the same well as the matched compound. Or
    # in the case of null_criteria_not, ensuring things do not match.
    null_criteria_query_list = []
    if len(null_criteria) > 0:
        tmp_matched_cpd_row = replicate_frame.iloc[0]
        for gbn in null_criteria:
            tmp_val = tmp_matched_cpd_row[gbn]
            if isinstance(tmp_val, (float, int)):
                null_criteria_query_list.append(f"{gbn}=={tmp_val}")
            else:
                null_criteria_query_list.append(f"{gbn}=='{tmp_val}'")
    # Generate non-matching queries
    if len(null_criteria_not) > 0:
        tmp_matched_cpd_row = replicate_frame.iloc[0]
        for gbn in null_criteria_not:
            tmp_val = tmp_matched_cpd_row[gbn]
            if isinstance(tmp_val, (float, int)):
                null_criteria_query_list.append(f"{gbn}!={tmp_val}")
            else:
                null_criteria_query_list.append(f"{gbn}!='{tmp_val}'")

    if isinstance(null_query_or_df, pd.DataFrame):
        if null_criteria_query_list!=[]:
            null_df = null_query_or_df.query(" and ".join(null_criteria_query_list))
        else:
            null_df=null_query_or_df
    else:
        if null_criteria_query_list!=[]:
            null_df = df.query(" and ".join(null_criteria_query_list)).query(null_query_or_df)
        else:
            null_df = df.query(null_query_or_df)


    # Calculate matched repeat score
    if similarity_metric == "spearman":
        matching_median_dist = np.median(
            np.corrcoef(np.argsort(np.argsort(replicate_frame[features].values)))[
                np.triu_indices(cardinality, k=1)
            ]
        )
    else:
        matching_median_dist = np.median(
            pdist(replicate_frame[features].values, metric=similarity_metric)
        )


    indices = [
        sample(range(null_df.shape[0]), cardinality)
        for _ in range(n_iters)]

    if len(indices) < 2:
        return _ReplicateData(matching_multiindex, len(matching_indices), np.nan, np.nan)

    # Extract the profiles then reahape to a 3D array n_iters x cardinality x features
    profiles = null_df[features].values[indices].reshape(n_iters, cardinality, len(features))

    # Calculate the null distribution
    if similarity_metric == "spearman":
        null_distribution = np.array(
            [
                np.median(
                    np.corrcoef(np.argsort(np.argsort(profiles[p])))[
                        np.triu_indices(cardinality, k=1)
                    ]
                )
                for p in range(n_iters)
            ]
        )
    else:
        null_distribution = np.array(
            [np.median(pdist(profiles[p], metric=similarity_metric)) for p in range(n_iters)]
        )

    return _ReplicateData(
        matching_multiindex,
        cardinality,
        matching_median_dist,
        null_distribution
        if return_full_profiles
        else np.percentile(null_distribution, float(percentile_cutoff)),
    )


def _setup_percent_replicating(
    ds: Union[Phenonaut, Dataset],
    features: Union[list[str], None],
    perturbation_column: Union[str, None],
    replicate_criteria: list[str],
    null_criteria: list[str],
    replicate_criteria_not: list[str],
    null_criteria_not: list[str],
) -> tuple[pd.DataFrame, list[str], Union[str, Callable], list[str], list[str]]:
    """Helper function to parse and set up variable for calculation of percent replicating

    If passed setup parameters are None, then an attempt to infer them is made.

    Parameters
    ----------
    ds : Union[Phenonaut, Dataset]
        Data
    features : Union[list[str], None]
        Dataset features
    perturbation_column : Union[str, None]
        Column in the dataframe used to denote unique perturbations.
    replicate_criteria : list[str]
        List of column headers to use to define matching replicates
    null_criteria : list[str]
        List of column headers to use to define matching parameters for the null
        distribution compounds.
    null_criteria_not : list[str]
        List of column headers to use to define non-matching parameters for the
        null distribution compounds.
    Returns
    -------
    tuple[pd.DataFrame, list[str], Union[str, Callable], list[str], list[str]]
        Tuple containing the required DataFrame, features, similarity_metric,
        replicate_criteria, null_criteria, and null_criteria_not.

    """
    if isinstance(ds, (Phenonaut, Dataset)):
        df, features, ret_pert_column = ds.get_df_features_perturbation_column(quiet=True)
        if ret_pert_column is not None and perturbation_column is None:
            perturbation_column = ret_pert_column
        if perturbation_column is None:
            raise ValueError(
                "No perturbation column found in Dataset, and the purturbation_column argument was None. Please pass it as an argument, or set on the Dataset - e.g.: ds.perturbation_column='pert_iname'"
            )
    elif isinstance(ds, pd.DataFrame):
        if features is None:
            raise ValueError("Must supply features if operating on a raw pd.DataFrame")
        df = ds
        if perturbation_column is None:
            raise ValueError(
                "No perturbation column set. Purturbation_column argument was None, and it is not discoverable from a pd.DataFrame. Please pass it as an argument"
            )
    else:
        raise ValueError(
            f"ds must be a Phenonaut object, a phenonaut.Dataset or a pd.DataFrame, it was {type(ds)}"
        )
    replicate_criteria = [] if replicate_criteria is None else replicate_criteria
    replicate_criteria = (
        [replicate_criteria] if isinstance(replicate_criteria, str) else replicate_criteria
    )
    replicate_criteria = (
        [perturbation_column] + replicate_criteria
        if perturbation_column not in replicate_criteria
        else replicate_criteria
    )

    null_criteria = [] if null_criteria is None else null_criteria
    null_criteria = [null_criteria] if isinstance(null_criteria, str) else null_criteria

    replicate_criteria_not = [] if replicate_criteria_not is None else replicate_criteria_not
    replicate_criteria_not = (
        [replicate_criteria_not]
        if isinstance(replicate_criteria_not, str)
        else replicate_criteria_not
    )

    null_criteria_not = [] if null_criteria_not is None else null_criteria_not
    null_criteria_not = (
        [null_criteria_not] if isinstance(null_criteria_not, str) else null_criteria_not
    )

    return (
        df,
        features,
        replicate_criteria,
        null_criteria,
        replicate_criteria_not,
        null_criteria_not,
    )


def percent_replicating(
    ds: Union[Dataset, Phenonaut, pd.DataFrame],
    perturbation_column: Union[str, None],
    replicate_query: Optional[str] = None,
    replicate_criteria: Optional[Union[str, list[str]]] = None,
    replicate_criteria_not: Optional[Union[str, list[str]]] = None,
    null_query_or_df: Optional[Union[str, pd.DataFrame]] = None,
    null_criteria: Optional[Union[str, list[str]]] = None,
    null_criteria_not: Optional[Union[str, list[str]]] = None,
    restrict_evaluation_query: Optional[str] = None,
    features: Optional[list[str]] = None,
    n_iters: int = 1000,
    similarity_metric: Union[str, Callable] = "spearman",
    similarity_metric_higher_is_better: bool = True,
    min_cardinality: int = 2,
    max_cardinality: int = 50,
    include_cardinality_violating_compounds_in_calculation: bool = False,
    return_full_performance_df: bool = False,
    additional_captured_params: Optional[dict] = None,
    similarity_metric_name: Optional[str] = None,
    performance_df_file: Optional[Union[str, Path]] = None,
    percentile_cutoff: int = 95,
    use_joblib_parallelisation: bool = True,
    n_jobs: int = -1,
):
    """Calculate percent replicating

    Percent replicating is defined by Way et. al. in:
    Way, Gregory P., et al. "Morphology and gene expression profiling provide complementary
    information for mapping cell state." Cell systems 13.11 (2022): 911-923.
    or on bioRxiv:
    https://www.biorxiv.org/content/10.1101/2021.10.21.465335v2

    Helpful descriptions also exist in
    https://github.com/cytomining/cytominer-eval/issues/21#issuecomment-902934931

    This implementation is designed to work with a variety of phenotypic similarity methods, not
    just a spearman correlation coefficient between observations.groupby_null

    Matching distributions are created by matching perturbations. In Phenonaut, this is
    typically defined by the perturbation_column field. This function takes this field as an
    argument, although it is unused if found in the passed Dataset/Phenonaut object. Additional
    criterial to match on such as concentration and well position can be added using the
    replicate_criteria argument. Null distributions are composed of a pick of C unique compounds,
    where C is the cardinality of the matched repeats (how many), and their median 'similarity'.
    By default, this similarity is the spearman correlation coefficient between profiles. This
    process to generate median similarity for non-replicate compounds is repeated n_iters times
    (by default 1000). Once the null distribution has been collected (median pairwise similarities),
    the median similarity of matched replicates is compared to the 95th percentile of this null
    distribution. If it is greater, then the compound (or compound and dose) are deemed replicating.
    Null distributions may not contain the matched compound. The percent replicating is calculated
    from the number of matched repeats which were replicating versus the number which were not.

    As the calculation is demanding, the function makes use of the joblib library for parallel
    calculation of the null distribution.

    Parameters
    ----------
    ds : Union[Dataset, Phenonaut, pd.DataFrame],
        Input data in the form of a Phenonaut dataset, a Phenonaut object, or a pd.DataFrame
        containing profiles of perturbations. If a Phenonaut object is supplied, then the last added
        Dataset will be used.
    perturbation_column : Union[str, None]
        In the standard % replicating calculation, compounds, are matched by name (or identifier),
        and dose, although this can be relaxed. This argument sets the column name
        containing an identifier for the perturbation name (or identifier), usually the name of a
        compound or similar. If a Phenonaut object or Dataset is supplied as the ds argument and this
        perturbation_column argument is None, then this value is attempted to be discovered through
        interrogation of the perturbation_column property of the Dataset. In the case of the CMAP_Level4
        PackagedDataset, a standard run would be achieved by passing 'pert_iname'as an argument here,
        or disregarding the value found in this argument by providing a dataset with the
        perturbation_column property already set.
    replicate_query: Optional[str]=None
        Optional pandas query to apply in selection of the matching replicates, this maybe something
        like ensuring concentration is above a threshold, or that they are taken from certain timepoints.
        Please note, if information in rows is never going to be included in matching or null distributions,
        then it is more efficient to prefilter the dataframe before running percent_replicating on it.
        This parameter should not be used to restrict the compounds on which percent replicating is run,
        as this is inefficient. Instead, the restrict_evaluation_query should be used.
    replicate_criteria : Optional[Union[str, list[str]]]=None
        As noted above describing the impact of the perturbation_column argument, matching compounds
        are often defined by their perturbation name/identifier and dose.  Whilst the perturbation
        column matches the compound name/identifier (something which must always be matched), passing
        a string here containing the title of a dose column (for example) also enforces matching on
        this property. A list of strings may also be passed. In the case of the PackagedDataset
        CMAP_Level4, this argument would take the value "pert_idose_uM" and would ensure that
        matched replicates share a common identifier/name (as default and enforced by
        perturbation_column) and concentration. The original perturbation_column may be included
        here but has no effect. By default None.
    replicate_criteria_not : Optional[Union[str, list[str]]]=None
        Values in this list enforce that matching replicates do NOT share a common property. This is
        useful for exotic evaluations, like picking replicates from across cell lines and
        concentrations.
    null_query_or_df : Optional[str, pd.DataFrame]=None
        Optional pandas query to apply in selection of the non-matching replicates comprising the null
        distribution. This can be things like ensuring only a certain plates or cell lines are used in
        construction of the distribution. Alternatively, a pd.DataFrame may be supplied here, from which
        the non-matching compounds are drawn for creation of the null distribution. Note; if
        supplying a query to filter out  information in rows that is never going to be included in
        matching or null distributions, then it is more efficient to prefilter the dataframe before
        running percent_replicating on it. This argument does not override null_criteria, or
        null_criteria_not as these have effect after this arguments effects have been applied. Has no
        effect if None. By default, None.
    null_criteria : Optional[Union[str, list[str]]]
        Whilst matching compounds are often defined by perturbation and dose, compounds comprising the
        null distribution must sometimes match the well position of the original compound. This
        argument captures the column name defining properties of non-matching replicates which must
        match the orignal query compound. In the case of the CMAP_Level4 PackagedDataset, this
        argument would take the value "well". A list of strings may also be passed to enforce further
        fields within the matching distribution which must match in the null distribution. If None,
        then no requirements appart from a different name/compound idenfier are enforced. By default
        None.
    null_criteria_not : Optional[Union[str, list[str]]]
        Values in this list enforce that matching non-replicates do NOT share a common property with
        the chosen replicates. The opposite of the above described null_criteria, this
        allows exotic evaluations like picking replicates from different cell lines to the matching
        replicates. Has no effect if None. By default None.
    restrict_evaluation_query: Optional[str], optional
        If only a few compounds in a Phenonaut Dataset are to be included in the percent replicating
        calculation, then this parameter may be used to efficiently select only the required compounds
        using a standard pandas style query which is run on groups defined by replicate_criteria
        Excluded compounds are not removed from the percent replicating calculation.  If None, then has
        no effect. By default None.
    features : features:Optional[list[str]]
        Features list which capture the phenotypic responses to perturbations. Only required and used
        if a pd.DataFrame is supplied in place of the ds argument. By default None.
    n_iters : int, optional
        Number of times the non-matching compound replicates should be sampled to compose the null
        distribution. If less than n_iters are available, then take as many as possible. By default
        1000.
    similarity_metric : Union[str, Callable, PhenotypicMetric], optional
        Callable metric, or string which is passed to pdist. This should be a
        distance metric; that is, lower is better, higher is worse.
        Note, a special case exists, whereby 'spearman' may be supplied here
        if so, then a much faster Numpy method np.corrcoef is used, and then results
        are subtracted from 1 to turn the metric into a distance metric. By default 'spearman'.
    similarity_metric_higher_is_better : bool
        If True, then a high value from the supplied similarity metric is better. If False, then
        a lower value is better (as is the case for distance metrics like Euclidean/Manhattan etc).
        Note that if lower is better, then the percentile should be changed to the other end of the
        distribution. For example, if keeping with significance at the 5 % level for a metric for
        which higher is better, then a metric where lower is better would use the 5th percentile,
        and percentile_cutoff = 5 should be used. By default True.
    min_cardinality : int
        Cardinality is the number of times a treatment is repeated (treatment with matching well,
        dose and any other constraints imposed). This arguments sets the minimum number of treatment
        repeats that should be present, if not, then the group is excluded from the calculation.
        Behavior of cytominer-eval includes all single repeat measuments, marking them as non-replicating
        this behaviour is replicated by setting this argument to 2. If 2, then only compounds with 2
        or more repeats are included in the calculation and have the posibility of generating a score
        for comparison to a null distribution and potentially passing the replicating test of being
        greater than the Nth percentile of the null distribution. By default 2.
    max_cardinality : int
        If a dataset has thousands of matched repeats, then little is gained in
        finding pairwise all-to-all distances of non-matching compounds, this
        argument allows setting an upper bound cutoff after which, the repeats
        are shuffled and max_cardinality samples drawn to create a synthetic set of
        max_cardinality repeats. This is very useful when using a demanding similarity
        method as it cuts the evaluations dramatically. By default 50.
    include_cardinality_violating_compounds_in_calculation : bool
        If True, then compounds for which there are no matching replicates, or not enough as defined
        by min_cardinality (and are therefore deemed not replicating) are included in the final
        reported percent replicating statistics. If False, then they are not included as
        non-replicating and simply ignored as if they were not present in the dataset. By default
        False.
    return_full_performance_df : bool
        If True, then a tuple is returned with the percent replicating score in the first position,
        and a pd.DataFrame containing full information on each repeat. By default False
    performance_df_file : Optional[Union[str, Path, bool]]
        If return_full_performance_df is True and a Path or str is given as an argument to this
        parameter, then the performance DataFrame is written out to a CSV file using this filename.
        If True is passed here, then the a filename will be constructed from function arguments,
        attempting to capture the run details. If an auto-generated file with this name exists,
        then an error is raised and no calculations are performed. In addition to the output CSV,
        a json file is also written capturing arguments that the function was called with. So if
        'pr_results.csv' is passed here, then a file named pr_results.json will be written out.
        If a filename is autogenerated, then the autogenerated filename is adapted to have the
        '.json' file extension. If the argument does not end in '.csv', then .json is appended
        to the end of the filename to define the name of the json file. By Default, None.
    additional_captured_params: Optional[dict]
        If writing out full details, also include this dictionary in the output json file, useful
        to add metadata to runs. By default None.
    similarity_metric_name : Optional[str]
        If relying on the function to make a nice performance CSV file name, then a nice succinct
        similarity metric name may be passed here, rather than relying upon calling __repr__ on the
        function, which may return long names such as:
        'bound method Phenotypic_Metric.similarity of Manhattan'. By default None.
    percentile_cutoff : int
        Percentile of the null distribution over which the matching replicates must score to be
        considered replicating. Should range from 0 to 100. By default 95.
    use_joblib_parallelisation : bool
        If True, then use joblib to parallelise evaluation of compounds. By default True.
    n_jobs : int, optional
        The n_jobs argument is passed to joblib for parallel execution and defines the number of
        threads to use.  A value of -1 denotes that the system should determine how many jobs to run.
        By default -1.

    Returns
    -------
    Union[float, tuple[float, pd.DataFrame]]
        If return_full_performance_df is False, then only the percent replicating is returned.
        If True, then a tuple is returned, with percent replicating in the first position, and a
        pd.DataFrame in the second position containing the median repeat scores, as well as median
        null distribution scores in an easy to analyse format.

    """
    start_time = timer()
    if max_cardinality < min_cardinality:
        raise ValueError(
            f"Error, max_cardinality ({max_cardinality}) may not be less than min_cardinality ({min_cardinality})"
        )
    if percentile_cutoff < 1:
        print(
            "WARNING: percentile_cutoff < 1. For the 95th percentile, this should be 95, not 0.95. This function calls np.percentile, which behaves the same, expecting a number between 0 and 100 to specify the cutoff."
        )
    (
        df,
        features,
        replicate_criteria,
        null_criteria,
        replicate_criteria_not,
        null_criteria_not,
    ) = _setup_percent_replicating(
        ds,
        features,
        perturbation_column,
        replicate_criteria,
        null_criteria,
        replicate_criteria_not,
        null_criteria_not,
    )

    if isinstance(similarity_metric, str):
        # Inbuilt fast methods, lookup table for higher_is_better
        special_similarity_metrics_higher_better_lookup = {
            "spearman": True,
            "euclidean": False,
            "cityblock": False,
            "cosine": False,
        }
        if similarity_metric in special_similarity_metrics_higher_better_lookup:
            if (
                special_similarity_metrics_higher_better_lookup[similarity_metric]
                != similarity_metric_higher_is_better
            ):
                print(
                    f"Warning, special similarity metric value {similarity_metric} passed with similarity_metric_higher_is_better = {similarity_metric_higher_is_better}, changing similarity_metric_higher_is_better to {special_similarity_metrics_higher_better_lookup[similarity_metric]}"
                )
                similarity_metric_higher_is_better = not similarity_metric_higher_is_better
        if similarity_metric_name is None:
            similarity_metric_name = similarity_metric

    if isinstance(similarity_metric, PhenotypicMetric):
        similarity_metric_higher_is_better = similarity_metric.higher_is_better
        similarity_metric_name = similarity_metric.name
        if similarity_metric.is_magic_string:
            similarity_metric = similarity_metric.func

    inspect_replicating_additional_args = {
        "df": df,
        "features": features,
        "replicate_query": replicate_query,
        "replicate_criteria": replicate_criteria,
        "replicate_criteria_not": replicate_criteria_not,
        "null_query_or_df": null_query_or_df,
        "null_criteria": null_criteria,
        "null_criteria_not": null_criteria_not,
        "similarity_metric": similarity_metric,
        "n_iters": n_iters,
        "min_cardinality": min_cardinality,
        "max_cardinality": max_cardinality,
        "return_full_profiles": return_full_performance_df,
        "percentile_cutoff": percentile_cutoff,
    }

    df_groupby_indices_and_items = (
        df.groupby(replicate_criteria).indices.items()
        if restrict_evaluation_query is None
        else df.query(restrict_evaluation_query).groupby(replicate_criteria).indices.items()
    )

    # Run in parallel
    if use_joblib_parallelisation:
        pr_results = Parallel(n_jobs=n_jobs)(
            delayed(_inspect_replicating)(
                **(inspect_replicating_additional_args), **{"g_indices": g_indices}
            )
            for g_indices in df_groupby_indices_and_items
            if (
                len(g_indices) >= min_cardinality
                or include_cardinality_violating_compounds_in_calculation
            )
        )
    # Or not
    else:
        pr_results = [
            _inspect_replicating(
                **(inspect_replicating_additional_args), **{"g_indices": g_indices}
            )
            for g_indices in progressbar(df_groupby_indices_and_items)
            if (
                len(g_indices) >= min_cardinality
                or include_cardinality_violating_compounds_in_calculation
            )
        ]

    # Remove compounds which did not make it through cardinality filters (as specified through args)
    if not include_cardinality_violating_compounds_in_calculation:
        pr_results = [r for r in pr_results if not np.isnan(r.matching_median_dist)]

    if len(pr_results) == 0:
        print(
            "No suitable compounds were found for calculation of any replicating compounds, reporting as such"
        )

    def _determine_replicating(
        null_dist, rep_val, similarity_metric_higher_is_better, percentile_cutoff
    ) -> int:
        if isinstance(null_dist, np.ndarray):
            null_dist = np.percentile(null_dist, percentile_cutoff)
        if np.isnan(null_dist):
            return 0
        if similarity_metric_higher_is_better:
            return 1 if rep_val > null_dist else 0
        else:
            return 1 if rep_val < null_dist else 0

    if len(pr_results) == 0:
        print(
            "No suitable compounds were found for calculation of any replicating compounds, reporting as such"
        )

    if return_full_performance_df:
        res_df = pd.DataFrame(
            data=[
                (
                    prd.cardinality,
                    _determine_replicating(
                        prd.null_nth_percentile_or_null_distribution,
                        prd.matching_median_dist,
                        similarity_metric_higher_is_better,
                        percentile_cutoff,
                    ),
                    prd.matching_median_dist,
                    np.percentile(prd.null_nth_percentile_or_null_distribution, percentile_cutoff),
                )
                for prd in pr_results
            ],
            columns=[
                "cardinality",
                "is_replicating",
                "median_replicate_score",
                f"null_{percentile_cutoff}_percentile",
            ],
            index=[prd.index for prd in pr_results],
        )
        null_d_df = pd.DataFrame(
            data=[
                prd.null_nth_percentile_or_null_distribution
                if isinstance(prd.null_nth_percentile_or_null_distribution, np.ndarray)
                else np.full(n_iters, np.nan)
                for prd in pr_results
            ],
            columns=[f"null_{ni}" for ni in range(1, n_iters + 1)],
            index=[prd.index for prd in pr_results],
        )
        res_df = pd.concat([res_df, null_d_df], axis=1)
        if performance_df_file is not None and performance_df_file != False:
            run_parameters_dict = {
                "ds.df.shape": df.shape,
                "perturbation_column": perturbation_column,
                "replicate_query": replicate_query,
                "replicate_criteria": replicate_criteria,
                "replicate_criteria_not": replicate_criteria_not,
                "null_query_or_df": null_query_or_df
                if not isinstance(null_query_or_df, pd.DataFrame)
                else null_query_or_df.shape,
                "null_criteria": null_criteria,
                "null_criteria_not": null_criteria_not,
                "features": features,
                "n_iters": n_iters,
                "similarity_metric": f"{similarity_metric}",
                "similarity_metric_higher_is_better": similarity_metric_higher_is_better,
                "similarity_metric_name": f"{similarity_metric}"
                if similarity_metric_name is None
                else similarity_metric_name,
                "min_cardinality": min_cardinality,
                "max_cardinality": max_cardinality,
                "include_cardinality_violating_compounds_in_calculation": include_cardinality_violating_compounds_in_calculation,
                "return_full_performance_df": return_full_performance_df,
                "percentile_cutoff": percentile_cutoff,
                "use_joblib_parallelisation": use_joblib_parallelisation,
                "n_jobs": n_jobs,
                "additional_captured_params": additional_captured_params,
            }

            performance_df_file = _check_performance_path_and_gen_if_needed(
                performance_df_file, run_parameters_dict, "pr"
            )
            run_parameters_dict[str(performance_df_file)] = str(performance_df_file)

            res_df.to_csv(performance_df_file)
            performance_json_file = (
                str(performance_df_file)[:-4]
                if str(performance_df_file).endswith(".csv")
                else str(performance_df_file)
            ) + ".json"
            run_parameters_dict["start_time"] = start_time
            run_parameters_dict["end_time"] = timer()
            run_parameters_dict["time_taken"] = (
                run_parameters_dict["end_time"] - run_parameters_dict["start_time"]
            )

            json.dump(run_parameters_dict, open(performance_json_file, "w"))

        if res_df.shape[0] == 0:
            return 0.0, res_df
        return (np.sum(res_df.is_replicating) / res_df.shape[0]) * 100, res_df

    else:
        if len(pr_results) == 0:
            return 0.0
        return (
            np.sum(
                [
                    _determine_replicating(
                        prd.null_nth_percentile_or_null_distribution,
                        prd.matching_median_dist,
                        similarity_metric_higher_is_better,
                        percentile_cutoff,
                    )
                    for prd in pr_results
                ]
            )
            / len(pr_results)
        ) * 100

























def percent_compact(
    ds: Union[Dataset, Phenonaut, pd.DataFrame],
    perturbation_column: Union[str, None],
    replicate_criteria: Optional[Union[str, list[str]]] = None,
    replicate_query: Optional[str] = None,
    replicate_criteria_not: Optional[Union[str, list[str]]] = None,
    null_query_or_df: Optional[Union[str, pd.DataFrame]] = None,
    null_criteria: Optional[Union[str, list[str]]] = None,
    null_criteria_not: Optional[Union[str, list[str]]] = None,
    restrict_evaluation_query: Optional[str] = None,
    features: Optional[list[str]] = None,
    n_iters: int = 1000,
    similarity_metric: Union[str, Callable] = "spearman",
    similarity_metric_higher_is_better: bool = True,
    min_cardinality: int = 2,
    max_cardinality: int = 50,
    include_cardinality_violating_compounds_in_calculation: bool = False,
    return_full_performance_df: bool = False,
    additional_captured_params: Optional[dict] = None,
    similarity_metric_name: Optional[str] = None,
    performance_df_file: Optional[Union[str, Path]] = None,
    percentile_cutoff: int = 95,
    use_joblib_parallelisation: bool = True,
    n_jobs: int = -1,
):
    """Calculate percent compact

    Compactness is defined by the spread of compound repeats compared to a randomly sampled
    background distribution.  For a given compound, its cardinality (num replicates), reffered
    to as C is determined. Then the median distance of all replicates is determined. This is then
    compared to a randomly sampled background. This background is obtained as follows: select C
    random compounds, calculate their median pairwise distances to each other, and store this.
    Repeat the process 1000 times and build a distribution of matched cardinality to the
    replicating compound.  The replicate treatments are deemed compact if its score is less than
    the 5th percentile of the background distribution (for distance metrics), and greater than
    the 95th percentile for similarity metrics.  Percent compact is simply the percentage of
    compounds which pass this compactness test.

    Matching distributions are created by matching perturbations. In Phenonaut, this is
    typically defined by the perturbation_column field. This function takes this field as an
    argument, although it is unused if found in the passed Dataset/Phenonaut object. Additional
    criterial to match on such as concentration and well position can be added using the
    replicate_criteria argument. Null distributions are composed of a pick of C unique compounds,
    where C is the cardinality of the matched repeats (how many), and their median 'similarity'.
    By default, this similarity is the spearman correlation coefficient between profiles. This
    process to generate median similarity for non-replicate compounds is repeated n_iters times
    (by default 1000).

    As the calculation is demanding, the function makes use of the joblib library for parallel
    calculation of the null distribution.

    Parameters
    ----------
    ds : Union[Dataset, Phenonaut, pd.DataFrame],
        Input data in the form of a Phenonaut dataset, a Phenonaut object, or a pd.DataFrame
        containing profiles of perturbations. If a Phenonaut object is supplied, then the last added
        Dataset will be used.
    perturbation_column : Union[str, None]
        This argument sets the column name containing an identifier for the perturbation name (or
        identifier), usually the name of a compound or similar. If a Phenonaut object or Dataset is
        supplied as the ds argument and this perturbation_column argument is None, then this value is
        attempted to be discovered through interrogation of the perturbation_column property of the
        Dataset. In the case of the CMAP_Level4 PackagedDataset, a standard run would be achieved by
        passing 'pert_iname'as an argument here, or disregarding the value found in this argument by
        providing a dataset with the perturbation_column property already set.
    replicate_criteria : Optional[Union[str, list[str]]]=None
        As noted above describing the impact of the perturbation_column argument, matching compounds
        are often defined by their perturbation name/identifier and dose.  Whilst the perturbation
        column matches the compound name/identifier (something which must always be matched), passing
        a string here containing the title of a dose column (for example) also enforces matching on
        this property. A list of strings may also be passed. In the case of the PackagedDataset
        CMAP_Level4, this argument would take the value "pert_idose_uM" and would ensure that
        matched replicates share a common identifier/name (as default and enforced by
        perturbation_column) and concentration. The original perturbation_column may be included
        here but has no effect. By default None.
    replicate_query: Optional[str]=None
        Optional pandas query to apply in selection of the matching replicates, this maybe something
        like ensuring concentration is above a threshold, or that they are taken from certain timepoints.
        Please note, if information in rows is never going to be included in matching or null distributions,
        then it is more efficient to prefilter the dataframe before running compactness on it.
        This parameter should not be used to restrict the compounds on which compactness is run,
        as this is inefficient. Instead, the restrict_evaluation_query should be used.
    replicate_criteria_not : Optional[Union[str, list[str]]]=None
        Values in this list enforce that matching replicates do NOT share a common property. This is
        useful for exotic evaluations, like picking replicates from across cell lines and
        concentrations.
    null_query_or_df : Optional[str, pd.DataFrame]=None
        Optional pandas query to apply in selection of the non-matching replicates comprising the null
        distribution. This can be things like ensuring only a certain plates or cell lines are used in
        construction of the distribution. Alternatively, a pd.DataFrame may be supplied here, from which
        the non-matching compounds are drawn for creation of the null distribution. Note; if
        supplying a query to filter out information in rows that is never going to be included in
        matching or null distributions, then it is more efficient to prefilter the dataframe before
        running compactness on it. This argument does not override null_criteria, or
        null_criteria_not as these have effect after this arguments effects have been applied. Has no
        effect if None. By default, None.
    null_criteria : Optional[Union[str, list[str]]]
        Whilst matching compounds are often defined by perturbation and dose, compounds comprising the
        null distribution must sometimes match the well position of the original compound. This
        argument captures the column name defining properties of non-matching replicates which must
        match the orignal query compound. In the case of the CMAP_Level4 PackagedDataset, this
        argument would take the value "well". A list of strings may also be passed to enforce further
        fields within the matching distribution which must match in the null distribution. If None,
        then no requirements appart from a different name/compound idenfier are enforced. By default
        None.
    null_criteria_not : Optional[Union[str, list[str]]]
        Values in this list enforce that matching non-replicates do NOT share a common property with
        the chosen replicates. The opposite of the above described null_criteria, this
        allows exotic evaluations like picking replicates from different cell lines to the matching
        replicates. Has no effect if None. By default None.
    restrict_evaluation_query: Optional[str], optional
        If only a few compounds in a Phenonaut Dataset are to be included in the compactness
        calculation, then this parameter may be used to efficiently select only the required compounds
        using a standard pandas style query which is run on groups defined by replicate_criteria
        Excluded compounds are not removed from the compactness calculation.  If None, then has
        no effect. By default None.
    features : features:Optional[list[str]]
        Features list which capture the phenotypic responses to perturbations. Only required and used
        if a pd.DataFrame is supplied in place of the ds argument. By default None.
    n_iters : int, optional
        Number of times the non-matching compound replicates should be sampled to compose the null
        distribution. If less than n_iters are available, then take as many as possible. By default
        1000.
    similarity_metric : Union[str, Callable, PhenotypicMetric], optional
        Callable metric, or string which is passed to pdist. This should be a
        distance metric; that is, lower is better, higher is worse.
        Note, a special case exists, whereby 'spearman' may be supplied here
        if so, then a much faster Numpy method np.corrcoef is used, and then results
        are subtracted from 1 to turn the metric into a distance metric. By default 'spearman'.
    similarity_metric_higher_is_better : bool
        If True, then a high value from the supplied similarity metric is better. If False, then
        a lower value is better (as is the case for distance metrics like Euclidean/Manhattan etc).
        Note that if lower is better, then the percentile should be changed to the other end of the
        distribution. For example, if keeping with significance at the 5 % level for a metric for
        which higher is better, then a metric where lower is better would use the 5th percentile,
        and percentile_cutoff = 5 should be used. By default True.
    min_cardinality : int
        Cardinality is the number of times a treatment is repeated (treatment with matching well,
        dose and any other constraints imposed). This arguments sets the minimum number of treatment
        repeats that should be present, if not, then the group is excluded from the calculation.
        Behavior of cytominer-eval includes all single repeat measuments, marking them as non-replicating
        this behaviour is replicated by setting this argument to 2. If 2, then only compounds with 2
        or more repeats are included in the calculation and have the posibility of generating a score
        for comparison to a null distribution and potentially passing the compactness test of being
        greater than the Nth percentile of the null distribution. By default 2.
    max_cardinality : int
        If a dataset has thousands of matched repeats, then little is gained in
        finding pairwise all-to-all distances of non-matching compounds, this
        argument allows setting an upper bound cutoff after which, the repeats
        are shuffled and max_cardinality samples drawn to create a synthetic set of
        max_cardinality repeats. This is very useful when using a demanding similarity
        method as it cuts the evaluations dramatically. By default 50.
    include_cardinality_violating_compounds_in_calculation : bool
        If True, then compounds for which there are no matching replicates, or not enough as defined
        by min_cardinality (and are therefore deemed not compact) are included in the final
        reported compactness statistics. If False, then they are not included as
        non-compact and simply ignored as if they were not present in the dataset. By default
        False.
    return_full_performance_df : bool
        If True, then a tuple is returned with the compactness score in the first position,
        and a pd.DataFrame containing full information on each repeat. By default False
    performance_df_file : Optional[Union[str, Path, bool]]
        If return_full_performance_df is True and a Path or str is given as an argument to this
        parameter, then the performance DataFrame is written out to a CSV file using this filename.
        If True is passed here, then the a filename will be constructed from function arguments,
        attempting to capture the run details. If an auto-generated file with this name exists,
        then an error is raised and no calculations are performed. In addition to the output CSV,
        a json file is also written capturing arguments that the function was called with. So if
        'compactness_results.csv' is passed here, then a file named compactness_results.json will
        be written out. If a filename is autogenerated, then the autogenerated filename is adapted
        to have the '.json' file extension. If the argument does not end in '.csv', then .json is appended
        to the end of the filename to define the name of the json file. By Default, None.
    additional_captured_params: Optional[dict]
        If writing out full details, also include this dictionary in the output json file, useful
        to add metadata to runs. By default None.
    similarity_metric_name : Optional[str]
        If relying on the function to make a nice performance CSV file name, then a nice succinct
        similarity metric name may be passed here, rather than relying upon calling __repr__ on the
        function, which may return long names such as:
        'bound method Phenotypic_Metric.similarity of Manhattan'. By default None.
    percentile_cutoff : int
        Percentile of the null distribution over which the matching replicates must score to be
        considered compact. Should range from 0 to 100. By default 95.
    use_joblib_parallelisation : bool
        If True, then use joblib to parallelise evaluation of compounds. By default True.
    n_jobs : int, optional
        The n_jobs argument is passed to joblib for parallel execution and defines the number of
        threads to use.  A value of -1 denotes that the system should determine how many jobs to run.
        By default -1.

    Returns
    -------
    Union[float, tuple[float, pd.DataFrame]]
        If return_full_performance_df is False, then only the percent compact statistic is returned.
        If True, then a tuple is returned, with percent compact in the first position, and a
        pd.DataFrame in the second position containing the median repeat scores, as well as median
        null distribution scores in an easy to analyse format.

    """
    start_time = timer()
    if max_cardinality < min_cardinality:
        raise ValueError(
            f"Error, max_cardinality ({max_cardinality}) may not be less than min_cardinality ({min_cardinality})"
        )
    if percentile_cutoff < 1:
        print(
            "WARNING: percentile_cutoff < 1. For the 95th percentile, this should be 95, not 0.95. This function calls np.percentile, which behaves the same, expecting a number between 0 and 100 to specify the cutoff."
        )
    (
        df,
        features,
        replicate_criteria,
        null_criteria,
        replicate_criteria_not,
        null_criteria_not,
    ) = _setup_percent_replicating(
        ds,
        features,
        perturbation_column,
        replicate_criteria,
        null_criteria,
        replicate_criteria_not,
        null_criteria_not,
    )

    if isinstance(similarity_metric, str):
        # Inbuilt fast methods, lookup table for higher_is_better
        special_similarity_metrics_higher_better_lookup = {
            "spearman": True,
            "euclidean": False,
            "cityblock": False,
            "cosine": False,
        }
        if similarity_metric in special_similarity_metrics_higher_better_lookup:
            if (
                special_similarity_metrics_higher_better_lookup[similarity_metric]
                != similarity_metric_higher_is_better
            ):
                print(
                    f"Warning, special similarity metric value {similarity_metric} passed with similarity_metric_higher_is_better = {similarity_metric_higher_is_better}, changing similarity_metric_higher_is_better to {special_similarity_metrics_higher_better_lookup[similarity_metric]}"
                )
                similarity_metric_higher_is_better = not similarity_metric_higher_is_better
        if similarity_metric_name is None:
            similarity_metric_name = similarity_metric

    if isinstance(similarity_metric, PhenotypicMetric):
        similarity_metric_higher_is_better = similarity_metric.higher_is_better
        similarity_metric_name = similarity_metric.name
        if similarity_metric.is_magic_string:
            similarity_metric = similarity_metric.func

    inspect_replicating_additional_args = {
        "df": df,
        "features": features,
        "replicate_query": replicate_query,
        "replicate_criteria": replicate_criteria,
        "replicate_criteria_not": replicate_criteria_not,
        "null_query_or_df": null_query_or_df,
        "null_criteria": null_criteria,
        "null_criteria_not": null_criteria_not,
        "similarity_metric": similarity_metric,
        "n_iters": n_iters,
        "min_cardinality": min_cardinality,
        "max_cardinality": max_cardinality,
        "return_full_profiles": return_full_performance_df,
        "percentile_cutoff": percentile_cutoff,
    }

    df_groupby_indices_and_items = (
        df.groupby(replicate_criteria).indices.items()
        if restrict_evaluation_query is None
        else df.query(restrict_evaluation_query).groupby(replicate_criteria).indices.items()
    )

    # Run in parallel
    if use_joblib_parallelisation:
        compactness_results = Parallel(n_jobs=n_jobs)(
            delayed(_inspect_replicating)(
                **(inspect_replicating_additional_args), **{"g_indices": g_indices}
            )
            for g_indices in df_groupby_indices_and_items
            if (
                len(g_indices) >= min_cardinality
                or include_cardinality_violating_compounds_in_calculation
            )
        )
    # Or not
    else:
        compactness_results = [
            _inspect_replicating(
                **(inspect_replicating_additional_args), **{"g_indices": g_indices}
            )
            for g_indices in progressbar(df_groupby_indices_and_items)
            if (
                len(g_indices) >= min_cardinality
                or include_cardinality_violating_compounds_in_calculation
            )
        ]

    # Remove compounds which did not make it through cardinality filters (as specified through args)
    if not include_cardinality_violating_compounds_in_calculation:
        compactness_results = [r for r in compactness_results if not np.isnan(r.matching_median_dist)]

    if len(compactness_results) == 0:
        print(
            "No suitable compounds were found for calculation of any replicating compounds, reporting as such"
        )

    def _determine_compact(
        null_dist, rep_val, similarity_metric_higher_is_better, percentile_cutoff
    ) -> int:
        if isinstance(null_dist, np.ndarray):
            null_dist = np.percentile(null_dist, percentile_cutoff)
        if np.isnan(null_dist):
            return 0
        if similarity_metric_higher_is_better:
            return 1 if rep_val > null_dist else 0
        else:
            return 1 if rep_val < null_dist else 0

    if len(compactness_results) == 0:
        print(
            "No suitable compounds were found for calculation of any replicating compounds, reporting as such"
        )

    if return_full_performance_df:
        res_df = pd.DataFrame(
            data=[
                (
                    prd.cardinality,
                    _determine_compact(
                        prd.null_nth_percentile_or_null_distribution,
                        prd.matching_median_dist,
                        similarity_metric_higher_is_better,
                        percentile_cutoff,
                    ),
                    prd.matching_median_dist,
                    np.percentile(prd.null_nth_percentile_or_null_distribution, percentile_cutoff),
                )
                for prd in compactness_results
            ],
            columns=[
                "cardinality",
                "is_compact",
                "median_replicate_score",
                f"null_{percentile_cutoff}_percentile",
            ],
            index=[prd.index for prd in compactness_results],
        )
        null_d_df = pd.DataFrame(
            data=[
                prd.null_nth_percentile_or_null_distribution
                if isinstance(prd.null_nth_percentile_or_null_distribution, np.ndarray)
                else np.full(n_iters, np.nan)
                for prd in compactness_results
            ],
            columns=[f"null_{ni}" for ni in range(1, n_iters + 1)],
            index=[prd.index for prd in compactness_results],
        )
        res_df = pd.concat([res_df, null_d_df], axis=1)
        if performance_df_file is not None and performance_df_file != False:
            run_parameters_dict = {
                "ds.df.shape": df.shape,
                "perturbation_column": perturbation_column,
                "replicate_query": replicate_query,
                "replicate_criteria": replicate_criteria,
                "replicate_criteria_not": replicate_criteria_not,
                "null_query_or_df": null_query_or_df
                if not isinstance(null_query_or_df, pd.DataFrame)
                else null_query_or_df.shape,
                "null_criteria": null_criteria,
                "null_criteria_not": null_criteria_not,
                "features": features,
                "n_iters": n_iters,
                "similarity_metric": f"{similarity_metric}",
                "similarity_metric_higher_is_better": similarity_metric_higher_is_better,
                "similarity_metric_name": f"{similarity_metric}"
                if similarity_metric_name is None
                else similarity_metric_name,
                "min_cardinality": min_cardinality,
                "max_cardinality": max_cardinality,
                "include_cardinality_violating_compounds_in_calculation": include_cardinality_violating_compounds_in_calculation,
                "return_full_performance_df": return_full_performance_df,
                "percentile_cutoff": percentile_cutoff,
                "use_joblib_parallelisation": use_joblib_parallelisation,
                "n_jobs": n_jobs,
                "additional_captured_params": additional_captured_params,
            }

            performance_df_file = _check_performance_path_and_gen_if_needed(
                performance_df_file, run_parameters_dict, "pc"
            )
            run_parameters_dict[str(performance_df_file)] = str(performance_df_file)

            res_df.to_csv(performance_df_file)
            performance_json_file = (
                str(performance_df_file)[:-4]
                if str(performance_df_file).endswith(".csv")
                else str(performance_df_file)
            ) + ".json"
            run_parameters_dict["start_time"] = start_time
            run_parameters_dict["end_time"] = timer()
            run_parameters_dict["time_taken"] = (
                run_parameters_dict["end_time"] - run_parameters_dict["start_time"]
            )

            json.dump(run_parameters_dict, open(performance_json_file, "w"))

        if res_df.shape[0] == 0:
            return 0.0, res_df
        return (np.sum(res_df.is_compact) / res_df.shape[0]) * 100, res_df

    else:
        if len(compactness_results) == 0:
            return 0.0
        return (
            np.sum(
                [
                    _determine_compact(
                        prd.null_nth_percentile_or_null_distribution,
                        prd.matching_median_dist,
                        similarity_metric_higher_is_better,
                        percentile_cutoff,
                    )
                    for prd in compactness_results
                ]
            )
            / len(compactness_results)
        ) * 100

def silhouette_score(
    ds: Union[Dataset, Phenonaut, pd.DataFrame],
    perturbation_column: Union[str, None],
    replicate_criteria: Optional[Union[str, list[str]]] = None,
    features: Optional[list[str]] = None,
    similarity_metric: Union[str, Callable] = "euclidean",
    similarity_metric_higher_is_better: bool = True,
    return_full_performance_df: bool = False,):


    if isinstance(similarity_metric, str):
        # Inbuilt fast methods, lookup table for higher_is_better
        special_similarity_metrics_higher_better_lookup = {
            "spearman": True,
            "euclidean": False,
            "cityblock": False,
            "cosine": False,
        }
        if similarity_metric in special_similarity_metrics_higher_better_lookup:
            if (
                special_similarity_metrics_higher_better_lookup[similarity_metric]
                != similarity_metric_higher_is_better
            ):
                print(
                    f"Warning, special similarity metric value {similarity_metric} passed with similarity_metric_higher_is_better = {similarity_metric_higher_is_better}, changing similarity_metric_higher_is_better to {special_similarity_metrics_higher_better_lookup[similarity_metric]}"
                )
                similarity_metric_higher_is_better = not similarity_metric_higher_is_better

    if isinstance(similarity_metric, PhenotypicMetric):
        similarity_metric_higher_is_better = similarity_metric.higher_is_better
        if similarity_metric.is_magic_string:
            similarity_metric = similarity_metric.func


    if isinstance(ds,Phenonaut):
        ds=ds[-1]
    if isinstance(ds, Dataset):
        df=ds.df
        features=ds.features

        if perturbation_column is None:
            if ds.perturbation_column is None:
                raise ValueError("No pertuabtion column set, and no perturbation_column argument given")
            perturbation_column=ds.perturbation_column
    else: # Must be df-like
        if perturbation_column is None:
            raise ValueError("Must provide a perturbation_column argument when supplying a DataFrame")
        if features is None:
            raise ValueError("Must provide features argument when supplying a DataFrame")
        df=df

    group_by=[perturbation_column]
    if replicate_criteria is not None:
        if isinstance(replicate_criteria, str):
            replicate_criteria=[replicate_criteria]
        group_by.extend(replicate_criteria)

    gb=df.groupby(group_by)
    n_clusters=len(gb)

    scores=np.eye(n_clusters)
    scores[scores==1]=np.nan


    for i, (group_name_a, indices_a) in enumerate(gb.indices.items()):
        for j, (group_name_b, indices_b) in enumerate(gb.indices.items()):
            if group_name_a==group_name_b:
                continue
            X=np.vstack([
            df.iloc[indices_a][features], df.iloc[indices_b][features]])
            labels=[*([group_name_a]*len(indices_a)), *([group_name_b]*len(indices_b))]
            scores[i,j]=sklearn_silhouette_score(X, labels, metric=similarity_metric)
    scores_df=pd.DataFrame(data=scores, index=list(gb.indices.keys()), columns=list(gb.indices.keys()))
    scores_df["mean_silhouette_score"]=np.nanmean(scores, axis=1)
    if return_full_performance_df:
        return scores_df['mean_silhouette_score'].mean(), scores_df
    else:
        return scores_df['mean_silhouette_score'].mean()

