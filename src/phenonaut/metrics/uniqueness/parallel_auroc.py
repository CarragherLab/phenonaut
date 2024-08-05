# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Optional
from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import os

_d = None
_metric = None


def _auroc_parallel_eval_phe_worker_init(
    big_data: np.ndarray,
    metric: PhenotypicMetric,
    max_cardinality: int,
    rng_seed: int = 7,
):
    global _d, _metric, _max_cardinality, _np_rng
    _d = big_data[:]
    _metric = deepcopy(metric)
    _max_cardinality = max_cardinality
    _np_rng = np.random.default_rng(rng_seed)
    os.sched_setaffinity(0, range(mp.cpu_count()))


def _auroc_parallel_eval_work(grp_key_indices_tuple):
    global _d
    global _metric
    global _max_cardinality
    global _np_rng
    d = _d[:]
    metric = deepcopy(_metric)
    if _max_cardinality is not None:
        if len(grp_key_indices_tuple[1]) > _max_cardinality:
            grp_key_indices_tuple = (
                grp_key_indices_tuple[0],
                _np_rng.choice(grp_key_indices_tuple[1], _max_cardinality),
            )
    scores = np.empty((len(grp_key_indices_tuple[1]), d.shape[0]))
    for i, q in enumerate(d[grp_key_indices_tuple[1]]):
        scores[i] = -metric.distance(q, d)
    y_true = np.zeros(d.shape[0], dtype=int)
    y_true[grp_key_indices_tuple[1]] = 1
    all_treatment_roc_scores = [roc_auc_score(y_true, s) for s in scores]
    results_dict = {
        "n_samples": len(scores),
        "trt_meanauroc": np.mean(all_treatment_roc_scores),
        "trt_aurocs": [all_treatment_roc_scores],
    }
    return grp_key_indices_tuple[0], results_dict


def auroc(
    ds: Union[Phenonaut, Dataset],
    groupby: Union[str, list[str]],
    phenotypic_metric: PhenotypicMetric,
    parallel: bool = True,
    allowed_pert_inames: Optional[list[str]] = None,
    quiet: bool = False,
    max_cardinality: int | None = None,
    random_state: int = 7,
) -> tuple[float | None, pd.DataFrame]:
    os.sched_setaffinity(0, range(1024))
    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    grouped_df = ds.df.groupby(groupby)

    grp_id_ilocs = grouped_df.indices.items()
    if allowed_pert_inames is not None:
        allowed_ilocs = np.argwhere(
            [v in allowed_pert_inames for v in ds.df.pert_iname.values]
        )
        grp_id_ilocs = [
            (gid, [il for il in grouped_df.indices[gid] if il in allowed_ilocs])
            for gid in grouped_df.indices
        ]

        grp_id_ilocs = [(gid, ils) for gid, ils in grp_id_ilocs if len(ils) > 0]
    group_label_list = list(grouped_df.indices.keys())
    y = np.empty(ds.df.shape[0], dtype=int)
    for group_name, indexes in grouped_df.indices.items():
        y[indexes] = group_label_list.index(group_name)
    res_df = pd.DataFrame()
    if len(set(y)) == 1:
        print("Only one y target found, AUROC requires multiple for calculation")
        return res_df
    if parallel:
        with mp.Pool(
            processes=None,
            initializer=_auroc_parallel_eval_phe_worker_init,
            initargs=(
                ds.data.values,
                phenotypic_metric,
                max_cardinality,
                random_state,
            ),
        ) as pool:
            for grp_name, res in tqdm(
                pool.imap_unordered(
                    _auroc_parallel_eval_work, grp_id_ilocs, chunksize=1
                ),
                total=len(grp_id_ilocs),
                desc="AUROC",
                disable=quiet,
            ):
                res_df = pd.concat(
                    [res_df, pd.DataFrame(res, index=[grp_name])], axis=0
                )
    else:
        for grp_key_indices_tuple in tqdm(grp_id_ilocs, desc="AUROC", disable=quiet):
            scores = np.empty((len(grp_key_indices_tuple[1]), ds.df.shape[0]))
            for i, q in enumerate(ds.data.values[grp_key_indices_tuple[1]]):
                scores[i] = -phenotypic_metric.distance(q, ds.data.values).flatten()
            y_true = np.zeros(ds.df.shape[0], dtype=int)
            y_true[grp_key_indices_tuple[1]] = 1
            all_treatment_roc_scores = [roc_auc_score(y_true, s) for s in scores]
            results_dict = {
                "n_samples": len(scores),
                "trt_meanauroc": np.mean(all_treatment_roc_scores),
                "trt_aurocs": [all_treatment_roc_scores],
            }
            res_df = pd.concat(
                [res_df, pd.DataFrame(results_dict, index=[grp_key_indices_tuple[0]])],
                axis=0,
            )
    return np.mean(res_df["trt_meanauroc"]) if len(res_df) > 0 else None, res_df
