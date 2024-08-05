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

__all__ = ["pertmutation_test_type2_distinct_from_query_group"]


def _permtest_FDR_parallel_eval_phe_worker_init(
    data, metric, rng_seed, n_iters
):
    global _data, _len_data, _metric,  _np_rng, _n_iters
    _data = data
    _len_data=len(data)
    _metric = metric
    _np_rng = np.random.default_rng(rng_seed + mp.current_process()._identity[0])
    _n_iters = n_iters
    os.sched_setaffinity(0, range(mp.cpu_count()))


def _permtest_FDR_parallel_eval_work(trt_indices):
    global _data, _len_data, _metric, _np_rng, _n_iters
    dmso_indices=np.setxor1d(trt_indices, range(_len_data))

    len_trt = len(trt_indices)
    len_veh = len(dmso_indices)-len_trt

    orig_dist = _metric(
        np.mean(_data[dmso_indices], axis=0),
        np.mean(_data[trt_indices], axis=0),
    )
    n_combinations = int(binom(len_veh + len_trt, len_trt))

    if n_combinations > _n_iters:
        generated_indexes = np.vstack(
            [
                _np_rng.choice(_len_data, size=len_trt, replace=False)
                for _ in range(_n_iters - 1)
            ],
            dtype=int,
        )
        n_gt_or_equal = 1
    else:
        generated_indexes = list(combinations(range(_len_data), len_trt))
        n_gt_or_equal = 0
    if len(generated_indexes)==0:
        return np.nan
    for indices in generated_indexes:
        np_indices=np.array(indices)
        d = _metric(
            np.mean(_data[np_indices], axis=0),
            np.mean(
                _data[np.setxor1d(np_indices, range(_len_data))], axis=0
            ),
        )
        if d >= orig_dist:
            n_gt_or_equal += 1
    return n_gt_or_equal / len(generated_indexes)


def pertmutation_test_type2_distinct_from_query_group(
    ds: Union[Phenonaut, Dataset, pd.DataFrame, np.ndarray],
    query_group_query: str | None,
    phenotypic_metric: PhenotypicMetric,
    cardinality:int=4,
    n_iters: int = 10000,
    n_trials: int = 1000,
    random_state: Union[int, np.random.Generator] = 42,
    max_samples_in_a_group=50,
    quiet: bool = False,
) -> tuple[float, pd.DataFrame | None]:
    # Manage RNG
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    dist_func = _phenotypic_metric_to_distance_function(phenotypic_metric)

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    if isinstance(ds, Dataset):
        if query_group_query is None:
            print(
                "Warning, no query_group_query, assuming all the dataset is to be used for FDR calculation"
            )
            ds = ds.data.values
        else:
            ds = ds.df.query(query_group_query)[ds.features].values
    elif isinstance(ds, pd.DataFrame):
        ds = ds.values
    else:
        if not isinstance(ds, np.ndarray):
            raise ValueError(
                "Could not process ds into an np.ndarray, please supply ds as one of: Phenonaut, Dataset,pd.DataFrame, np.ndarray"
            )

    if len(ds) == 0:
        raise ValueError(
            "No records found, if specifying a query, make sure it selects rows:"
            f" {query_group_query}",
        )

    # Downsample if too many samples
    if ds.shape[0] > max_samples_in_a_group:
        ds = ds[
            random_state.choice(len(ds), max_samples_in_a_group, replace=False)
        ]

    dummy_trt_indices=np.vstack(
            [
                random_state.choice(len(ds), size=cardinality, replace=True)
                for _ in range(n_trials)
            ],
            dtype=int,
        )

    pbar = tqdm(total=n_trials, disable=quiet, desc="Permutation test")
    p_vals=np.empty(len(dummy_trt_indices))
    p_val_i=0
    with mp.Pool(
        None,
        initializer=_permtest_FDR_parallel_eval_phe_worker_init,
        initargs=(
            ds,
            dist_func,
            random_state.choice(np.iinfo(np.int32).max),
            n_iters,
        ),
    ) as pool:
        for res in pool.imap_unordered(
            _permtest_FDR_parallel_eval_work, dummy_trt_indices, chunksize=1
        ):
            if not np.isnan(res):
                p_vals[p_val_i] = res
                p_val_i+=1
            pbar.update(1)
    pbar.close()
    return (np.sum(p_vals[:p_val_i] < 0.05, axis=0) / p_val_i) * 100

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
