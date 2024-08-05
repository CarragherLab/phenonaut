# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Callable
from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.special import binom
import multiprocessing as mp
import os

__all__ = ["pertmutation_test_distinct_from_query_group"]


def _permtest_parallel_eval_phe_worker_init(
    dataframe_values, dmso_indices, metric, rng_seed, max_samples_in_a_group, n_iters
):
    global _data, _metric, _dmso_indices, _np_rng, _max_samples_in_a_group, _n_iters
    _data = dataframe_values
    _metric = metric
    _dmso_indices = dmso_indices
    _np_rng = np.random.default_rng(rng_seed + mp.current_process()._identity[0])
    _max_samples_in_a_group = max_samples_in_a_group
    _n_iters = n_iters
    os.sched_setaffinity(0, range(mp.cpu_count()))


def _permtest_parallel_eval_work(trt_info):
    global _data, _metric, _dmso_indices, _np_rng, _max_samples_in_a_group, _n_iters

    trt_name, trt_indices = trt_info
    if len(trt_indices) > _max_samples_in_a_group:
        trt_indices = _np_rng.choice(
            trt_indices, _max_samples_in_a_group, replace=False
        )

    len_veh = len(_dmso_indices)
    len_trt = len(trt_indices)
    
    # if len_trt<2:
    #     return trt_name, np.nan
    stack_indies=np.hstack([_dmso_indices, trt_indices])
    stacked_data = _data[stack_indies]
    orig_dist = _metric(
        np.mean(stacked_data[:len_veh], axis=0),
        np.mean(stacked_data[len_veh:], axis=0),
    )
    n_combinations = int(binom(len_veh + len_trt, len_trt))

    if n_combinations > _n_iters:
        generated_indexes = np.vstack(
            [
                _np_rng.choice(len(stacked_data), size=len_trt, replace=False)
                for _ in range(_n_iters - 1)
            ],
            dtype=int,
        )
        n_gt_or_equal = 1
    else:
        generated_indexes = list(combinations(range(len(stacked_data)), len_trt))
        n_gt_or_equal = 0
        _n_iters = n_combinations


    for indices in generated_indexes:
        d = _metric(
            np.mean(stacked_data[np.array(indices)], axis=0),
            np.mean(
                stacked_data[np.setxor1d(indices, range(len(stacked_data)))], axis=0
            ),
        )
        if d >= orig_dist:
            n_gt_or_equal += 1
    return trt_name, n_gt_or_equal / _n_iters

def pertmutation_test_distinct_from_query_group(
    ds: Union[Phenonaut, Dataset, list[pd.DataFrame]],
    query_group_query: str,
    groupby: Union[str, list[str], None],
    phenotypic_metric: PhenotypicMetric,
    n_iters=10000,
    return_full_results_df: bool = True,
    random_state: Union[int, np.random.Generator] = 42,
    max_samples_in_a_group=50,
    quiet: bool = False,
    no_error_on_empty_query: bool = True,
) -> tuple[float, pd.DataFrame | None]:
    # Manage RNG
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    dist_func = _phenotypic_metric_to_distance_function(phenotypic_metric)

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    if not isinstance(ds, Dataset):
        raise ValueError(
            f"ds was found to be of type '{type(ds)}', should be phenonaut.Phenonaut,"
            " phenonaut.Dataset, or list[pd.DataFrame] (of length 2)"
        )
    
    veh_indices = ds.df.reset_index().index.get_indexer(ds.df.reset_index().query(query_group_query).index)
    if len(veh_indices) == 0:
        if no_error_on_empty_query:
            return 0.0, None
        raise ValueError(
            "No records found for the query group using the query:"
            f" {query_group_query}",
        )

    ptest_res_df = pd.DataFrame()

    grouped_df_id_to_iloc_tuples = list(ds.df.groupby(groupby).indices.items())

    if veh_indices.shape[0] > max_samples_in_a_group:
        veh_indices = random_state.choice(
            veh_indices, max_samples_in_a_group, replace=False
        )

    pbar = tqdm(total=len(grouped_df_id_to_iloc_tuples), disable=quiet, desc="Permutation test")
    with mp.Pool(
        None,
        initializer=_permtest_parallel_eval_phe_worker_init,
        initargs=(
            ds.data.values,
            veh_indices,
            dist_func,
            random_state.choice(np.iinfo(np.int32).max),
            max_samples_in_a_group,
            n_iters,
        ),
    ) as pool:
        for res in pool.imap_unordered(
            _permtest_parallel_eval_work,grouped_df_id_to_iloc_tuples, chunksize=1
        ):
            ptest_res_df.loc[str(res[0]), "pval"] = res[1]
            pbar.update(1)
    pbar.close()
    percent_distinct = ((np.sum(ptest_res_df < 0.05, axis=0) / len(ptest_res_df)) * 100).item()

    if return_full_results_df:
        return percent_distinct, ptest_res_df
    else:
        return percent_distinct


def _phenotypic_metric_to_distance_function(
    phenotypic_metric: PhenotypicMetric,
) -> Callable:
    """Get distance function from metric which could contain magic values

    Metrics may have "spearman" etc as their distance function. This can cause issues when a
    distance function is needed, and so this function returns a distance function for any
    PhenotypicMetric with a string for the distance function

    Parameters
    ----------
    phenotypic_metric : PhenotypicMetric
        PhenotypicMetric which could contain a string for the .func field.  It could also be a
        similarity metric. This function returns a distance function.

    Returns
    -------
    function
        Distance function
    """

    if isinstance(phenotypic_metric.func, str):
        if (
            phenotypic_metric.func.lower() == "spearman"
            or phenotypic_metric.func.lower() == "rank"
        ):

            def dist_func(x, y):
                return 1 - np.corrcoef(np.argsort(np.argsort(np.vstack([x, y]))))[0, 1]
        else:
            if phenotypic_metric.higher_is_better:

                def dist_func(x, y):
                    return (
                        1
                        - squareform(
                            pdist(np.vstack(x, y), metric=phenotypic_metric.func)
                        )[0, 1]
                    )
            else:

                def dist_func(x, y):
                    squareform(pdist(np.vstack([x, y]), metric=phenotypic_metric.func))[
                        0, 1
                    ]
    else:
        dist_func = phenotypic_metric.distance
    return dist_func
