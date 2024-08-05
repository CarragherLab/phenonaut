import phenonaut
from .compactness import percent_replicating
from .distinctness import (
    pertmutation_test_distinct_from_query_group,
    pertmutation_test_type2_distinct_from_query_group,
)
from .uniqueness import auroc
from .non_ds_phenotypic_metrics import PhenotypicMetric, non_ds_phenotypic_metrics
import numpy as np
import logging
from pathlib import Path

__all__ = ["run_cdu_benchmarks", "write_cdu_json"]


def run_cdu_benchmarks(
    ds: phenonaut.data.Dataset | phenonaut.Phenonaut,
    replicate_groupby: str | list[str] | None = None,
    dmso_query: str | None = None,
    run_percent_replicating: bool = True,
    run_permutation_test_against_dmso: bool = True,
    run_auroc: bool = True,
    run_distinct_type2: bool = False,
    return_full_results: bool = False,
    phenotypic_metric: PhenotypicMetric | str = non_ds_phenotypic_metrics["Euclidean"],
    quiet: bool = False,
    random_state: int = 42,
    max_cardinality: int = 50,
    permtest_iterations: int = 10000,
    pctrep_iterations: int = 1000,
    permtest_type2_trials: int = 1000,
    permtest_type2_cardinality: int | None = None,
    json_serialize_dataframes: bool = False,
) -> dict:
    """Run Compactness, Distinctness and Uniquness benchmarks

    Run 3 assay benchmarks to evaluate compactness (with percent replicating), distinctness (
    with a permutation test of all replicates against DMSO), and uniqueness (AUROC). Results
    are returned in a dictionary with the keys 'compactness_percent_replicating',
    'distinctness_percent_permutation_test', and 'uniqueness_auroc_mean'.  If the
    return_full_results parameter is true, then three additional items are added to the dataset
    under the keys 'percent_replicating_df', 'percent_distinct_df', and 'uniqueness_auroc_df'. These
    items contain dataframes giving a full breakdown of each replicate group within each test.


    Parameters
    ----------
    ds : phenonaut.data.Dataset | phenonaut.Phenonaut
        The dataset upon which the CDU metrics will be run. Can be a phenonaut.data.Dataset, or a
        Phenonaut object (in which case the last added dataset will be used).
    replicate_groupby : str | list[str] | None, optional
        Column name which may be used to define replicate groups. Can be a list of multiple column
        names which are then used in a pandas style groupby to define replicate groups. If None,
        then the perturbation_column of the dataset is used to define replicate groups., by default
        None
    dmso_query : str | None, optional
       A pandas style query which can be used to define the DMSO replicate group. If None, then an
       attempt to generate the query using the dataset's perturbation_column is used, extracting
       rows with 'DMSO' in the column, by default None
    run_percent_replicating : bool, optional
        If True, then percent replicating is run on the dataset to determine the percentage of
        treatments which are deemed distinct, by default True
    run_permutation_test_against_dmso : bool, optional
        If True, then the permutation test is run on the dataset to calculate the percentage of
        treatments which are distinct from DMSO, by default True
    run_auroc : bool, optional
        If True, then the AUROC test is run on the dataset to determine uniqueness, by default True
    run_distinct_type2 : bool, optional
        If True, then the distinctness false discovery rate is calculated, by default False
    return_full_results : bool, optional
        If True, then dataframes breaking down replicate group performance are included in the
        returned dictionary, by default False
    phenotypic_metric : PhenotypicMetric | str, optional
        The distance metric which should be used to compair treatments. This can be a
        PhenotypicMetric object, or one of: 'Connectivity', 'Rank', 'Zhang', 'Euclidean',
        'Manhattan', or 'Cosine', by default "Euclidean".
    quiet : bool, optional
        If True, then do not display progressbars, by default False
    random_state: int
        The integer which should be used to seed random number generators used in calculation of CDU
        metrics. By default 42
    max_cardinality : int, optional
        Maximum replicate group cardinality. If more than this limit, then the group will be
        downsampled, by default 50
    permtest_iterations : int
        The number of iterations used in the permutation test, this number of group permutations
        will be used in determination of P values. By default 10000
    permtest_type2_trials : int
        The number of times a subset of DMSO is held out and tested for membership in the DMSO
        population in order to approximate the type2 error, by default 100
    permtest_type2_cardinality : int | None
        The cardinality of held out dummy treatment DMSO groups. If None, then the most common
        cardinality in the dataset is determined and used, by default None
    pctrep_iterations : int
        The number of iterations use in the percent replicating calculation. This many null or
        background distributions (non-replicates) will be constructed for comparison against
        median inter-replicate group distances. By default 1000
    json_serialize_dataframes : bool, optional
        If True, then pandas DataFrames returned in results dictionaries are JSON serialized, by
        default False

    Returns
    -------
    dict
        Dictionary containing CDU metric performance information
    """
    if isinstance(phenotypic_metric, str):
        if phenotypic_metric in non_ds_phenotypic_metrics.keys():
            phenotypic_metric = non_ds_phenotypic_metrics[phenotypic_metric]
        else:
            raise ValueError(
                f"Could not find metric named '{phenotypic_metric}' in the keys of non_ds_phenotypic_metrics"
            )
    if isinstance(ds, phenonaut.Phenonaut):
        if len(ds.datasets) > 1:
            raise NotImplementedError(
                "run_cdu_benchmarks cannot integrate data (more than one dataset found in phenonaut object).  Please use other phenonaut functionality to integrate the data first."
            )
        ds = ds.datasets[0]

    if isinstance(ds, list):
        if len(ds) > 1:
            raise NotImplementedError(
                "Multiview benchmarking not currently implemented"
            )
        if len(ds) == 1:
            ds = ds[0]
    if not isinstance(ds, phenonaut.data.Dataset):
        raise TypeError(
            "ds could not be coerced into a single phenonaut.data.Dataset, the type of ds was",
            type(ds),
        )

    if replicate_groupby is None:
        replicate_groupby = ds.perturbation_column
    if isinstance(replicate_groupby, str):
        replicate_groupby = [replicate_groupby]
    if dmso_query is None:
        if len(replicate_groupby) > 1:
            dmso_query = f"{replicate_groupby[0]} == 'DMSO'"
            logging.info(
                f"No DMSO query specified, and more than one perturbation_column, guessing that the appropriate query is: {dmso_query}"
            )
        else:
            dmso_query = f"{ds.perturbation_column} == 'DMSO'"
            logging.info(
                f"No DMSO query specified, guessing that the appropriate query is: {dmso_query}"
            )

    results_dict = {}

    if run_percent_replicating:
        pr, pr_df = percent_replicating(
            ds,
            return_full_performance_df=True,
            phenotypic_metric=phenotypic_metric,
            max_cardinality=max_cardinality,
            random_state=random_state,
            n_iters=pctrep_iterations,
            parallel=True,
            quiet=quiet,
        )
        results_dict["compactness_percent_replicating"] = pr
        if return_full_results:
            if json_serialize_dataframes:
                results_dict["percent_replicating_df"] = (
                    pr_df.to_json() if pr_df is not None else None
                )
            else:
                results_dict["percent_replicating_df"] = pr_df

    if run_permutation_test_against_dmso:
        (
            percent_distinct,
            perm_test_results,
        ) = pertmutation_test_distinct_from_query_group(
            ds,
            dmso_query,
            ds.perturbation_column,
            phenotypic_metric=phenotypic_metric,
            max_samples_in_a_group=max_cardinality,
            quiet=quiet,
            random_state=random_state,
            n_iters=permtest_iterations,
        )
        results_dict["distinctness_percent_permutation_test"] = (
            np.nan if percent_distinct is None else percent_distinct
        )
        if return_full_results:
            if json_serialize_dataframes:
                results_dict["percent_distinct_df"] = (
                    perm_test_results.to_json()
                    if perm_test_results is not None
                    else None
                )
            else:
                results_dict["percent_distinct_df"] = perm_test_results

    if run_distinct_type2:
        if permtest_type2_cardinality is None:
            permtest_type2_cardinality = sorted(
                [
                    (k, v)
                    for k, v in ds.df.groupby(ds.perturbation_column)[
                        ds.perturbation_column
                    ]
                    .agg("count")
                    .value_counts()
                    .items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]

        type2_error = pertmutation_test_type2_distinct_from_query_group(
            ds,
            dmso_query,
            phenotypic_metric=phenotypic_metric,
            cardinality=permtest_type2_cardinality,
            quiet=quiet,
            random_state=random_state,
            n_iters=permtest_iterations,
            n_trials=permtest_type2_trials,
        )
        results_dict["distinctness_percent_type2_permutation_test"] = (
            np.nan if type2_error is None else type2_error
        )

    if run_auroc:
        mean_auroc, auroc_results = auroc(
            ds,
            ds.perturbation_column,
            phenotypic_metric=phenotypic_metric,
            random_state=random_state,
            max_cardinality=max_cardinality,
            quiet=quiet,
        )
        results_dict["uniqueness_auroc_mean"] = mean_auroc
        if return_full_results:
            if json_serialize_dataframes:
                results_dict["uniqueness_auroc_df"] = (
                    auroc_results.to_json() if auroc_results is not None else None
                )
            else:
                results_dict["uniqueness_auroc_df"] = auroc_results
    return results_dict


def write_cdu_json(
    results_dict: dict,
    output_path: str | Path,
    dataset_name: str,
    metric_name: str,
    test_splits: bool,
):
    import json
    from uuid import uuid4

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    json.dump(
        {
            "data_type": (
                "cdu_performance" if not test_splits else "cdu_performance_test_split"
            ),
            "dataset": dataset_name,
            "metric": f"{metric_name}",
            "scores": results_dict,
        },
        open(
            output_path / (uuid4().hex + ".json"),
            "w",
        ),
    )


def get_cdu_performance_df(cdu_dir: str | Path):
    import json
    import pandas as pd

    cdu_dir = Path(cdu_dir)
    if not cdu_dir.exists():
        raise FileNotFoundError(
            f"Requested CDU performance/results directory ({cdu_dir}) not found"
        )
    data = []
    for d in [json.load(open(f)) for f in cdu_dir.glob("*.json")]:
        view = "N/A"
        cell_line = "N/A"
        try:
            split_dataset_name = d["dataset"].split("_")
            view = split_dataset_name[2]
            cell_line = split_dataset_name[3]
        except:
            print(
                f"Trouble determining view and cell line info for database named '{d['dataset']}'"
            )
        d.update(
            {
                "Uniqueness (AUROC)": d["scores"]["uniqueness_auroc_mean"],
                "Compactness (% replicating)": d["scores"][
                    "compactness_percent_replicating"
                ],
                "Distinctness (% passing DMSO permutation test)": d["scores"][
                    "distinctness_percent_permutation_test"
                ],
                "View": view,
                "CellLine": cell_line,
                "Metric": d["metric"],
                "Dataset": d["dataset"],
            }
        )
        del d["scores"], d["metric"], d["dataset"]

        data.append(d)
    return pd.DataFrame(data)
