import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def percent_replicating_results_dataframe_to_percentile_vs_percent_replicating(
    df: pd.DataFrame,
    percentile_range: tuple[int, int] = (0, 101),
    percentile_step_size: int = 1,
    return_counts: bool = False,
    n_jobs: int = -1,
    similarity_metric_higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Get x,y arrays for cutoff vs % replicating plots

    Reads a DataFrame from phenonaut.metrics.performance.percent_replicating
    when run with return_full_performance_df = True, and generates a tuple of
    x and y coordinates, allowing plotting of percentile cutoff vs percent
    replicating.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame supplied by phenonaut.metrics.performance.percent_replicating
    percentile_range : tuple[int, int], optional
        The range of percentiles to cover, by default (0, 101)
    percentile_step_size : int, optional
        By default, every value in percentile_range is explored (-1 for the max
        value inkeeping with Python range function operation), the stepsize may
        be changed here. By default 1.
    return_counts : bool, optional
        If True, then y values denote the counts of replicates which were
        deemed replicating. If False, then percent replicating is returned. By default
        False
    similarity_metric_higher_is_better: bool
        If True, then consider treatment replicating if score is greater than
        the percentile cutoff.  If False, then consider treatment replicating if
        score is less than percentile cutoff. By default True.
    n_jobs : int, optional
        The n_jobs argument is passed to joblib for parallel execution and defines
        number of threads to use.  A value of -1 denotes that the system should
        determine how many jobs to run. By default -1.
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing 2 np.ndarrays, the first being percentile cutoff, and
        the second being mathching % replicating values (or count of replicating
        compounds if return_counts=True)
    """
    num_in_null = len(
        [
            1
            for nc in df.columns
            if nc.startswith("null_") and not nc.endswith("_percentile")
        ]
    )
    x = np.arange(*percentile_range)
    y = []

    def _assign_pr(r, p, similarity_metric_higher_is_better):
        if isinstance(r[-num_in_null], str):
            return 0
        if similarity_metric_higher_is_better:
            if r["median_replicate_score"] > np.percentile(r[-num_in_null:], p):
                return 1
            else:
                return 0
        else:
            if r["median_replicate_score"] < np.percentile(r[-num_in_null:], 100 - p):
                return 1
            else:
                return 0

    def _get_pr(df, p, similarity_metric_higher_is_better):
        return (
            p,
            np.sum(
                df.apply(
                    _assign_pr,
                    p=p,
                    axis=1,
                    similarity_metric_higher_is_better=similarity_metric_higher_is_better,
                )
            ),
        )

    if df.shape[0] == 0:
        return x, np.full(x.shape, np.nan)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_get_pr)(df, percentile, similarity_metric_higher_is_better)
        for percentile in x
    )

    sorted(results)
    y = np.array([r[1] for r in results])
    return (x, (y / df.shape[0]) * 100) if not return_counts else (x, y)


def percent_replicating_summarise_results(
    file_or_dir: Union[Path, str],
    dir_glob: str = "pr_*.csv",
    if_no_json_use_filename_to_derive_info: bool = True,
    get_percent_replicating: bool = True,
) -> Union[None, pd.DataFrame]:
    """Summarise percent replicating results

    Parameters
    ----------
    file_or_dir : Union[Path, str]
        Percent replicating results file, or directory containing results files.
    dir_glob : str, optional
        Glob to use in searching directories for percent replicating results
        files, by default "pr_*.csv".
    if_no_json_use_filename_to_derive_info : bool, optional
        Legacy runs of percent replicating did not produce a json information file
        and therefore run information is attempted to be derived from filenames
        if this parameter is True. By default True.
    get_percent_replicating : bool
        If True, then calculate and return PR in the table. Has no effect if
        run on a file (not a directory). By default True.
    Returns
    -------
    Union[Tuple[float, int], pd.DataFrame]
        Either a tuple containing the percent replicating and number of records
        contributing to that score (if run on a single file), or a pd.DataFrame
        summarising results (if run on a directory).
    """

    def _getpr_and_len(f):
        df = pd.read_csv(f)
        if df.shape[0] == 0:
            return 0.0, 0
        return (np.sum(df.is_replicating) / df.shape[0]) * 100, df.shape[0]

    file_or_dir = Path(file_or_dir)

    if not file_or_dir.exists():
        raise FileNotFoundError(f"{file_or_dir} not found")
    if file_or_dir.is_file():
        return _getpr_and_len(file_or_dir)

    elif file_or_dir.is_dir():
        files = file_or_dir.glob(dir_glob)
        standard_columns = [
            "filename",
            "similarity_metric",
            "similarity_metric_higher_is_better",
            "percent_replicating",
            "replicate_criteria",
            "replicate_criteria_not",
            "replicate_query_or_df",
            "null_criteria",
            "null_criteria_not",
            "null_query_or_df",
            "n_contributing",
        ]

        series_list = []
        for file in files:
            if get_percent_replicating:
                calculated_pr, len_df = _getpr_and_len(file)
            else:
                calculated_pr, len_df = (None, None)
            if file.with_suffix(".json").exists():
                d = json.load(open(file.with_suffix(".json")))
                columns = standard_columns.copy()
                new_data = [
                    str(file),
                    d["similarity_metric_name"],
                    d["similarity_metric_higher_is_better"],
                    calculated_pr,
                    d["replicate_criteria"],
                    d["replicate_criteria_not"],
                    d["replicate_query"],
                    d["null_criteria"],
                    d["null_criteria_not"] if d["null_criteria_not"] != [] else np.nan,
                    d["null_query_or_df"],
                    len_df,
                ]

                additional_data = d.get("additional_captured_params", None)
                if isinstance(additional_data, dict):
                    for k, v in additional_data.items():
                        columns.append(k)
                        new_data.append(v)
                series_list.append(pd.Series(data=new_data, index=columns))

            else:
                if if_no_json_use_filename_to_derive_info:
                    fstr = file.stem
                    if fstr.startswith("pr__"):
                        similarity_metric = "Spearman"
                    similarity_metric = fstr.split("__")[0].split("_")
                    cell_lines = fstr.split("__")[1]
                    matching = fstr.split("__matching")[1]
                    if "__" in matching:
                        matching = matching.split("__")[0]
                    nullmatch = (
                        ""
                        if "__nullmatch" not in fstr
                        else fstr.split("__nullmatch")[1]
                    )
                    if "__" in nullmatch:
                        nullmatch = nullmatch.split("__")[0]
                    columns = standard_columns.copy()
                    columns.append("CellLine")
                    new_data = [
                        str(file),
                        similarity_metric,
                        calculated_pr,
                        matching,
                        nullmatch,
                        len_df,
                        cell_lines,
                    ]
                    series_list.append(pd.Series(data=new_data, index=columns))
        if len(series_list) < 2:
            return pd.DataFrame(series_list).T
        return (
            pd.concat(series_list, axis=1)
            .T.set_index("filename")
            .sort_values("percent_replicating", ascending=False)
            .dropna(axis="columns")
        )


def percent_compact_summarise_results(
    file_or_dir: Union[Path, str],
    dir_glob: str = "pc_*.csv",
    get_percent_compact: bool = True,
) -> Union[None, pd.DataFrame]:
    """Summarise percent compact results

    Parameters
    ----------
    file_or_dir : Union[Path, str]
        Percent compact results file, or directory containing results files.
    dir_glob : str, optional
        Glob to use in searching directories for percent replicating results
        files, by default "pc_*.csv".
    get_percent_compact : bool
        If True, then calculate and return percent compact in the table. Has no
        effect if run on a file (not a directory). By default True.
    Returns
    -------
    Union[Tuple[float, int], pd.DataFrame]
        Either a tuple containing the percent compact and number of records
        contributing to that score (if run on a single file), or a pd.DataFrame
        summarising results (if run on a directory).
    """

    def _getpc_and_len(f):
        df = pd.read_csv(f)
        if df.shape[0] == 0:
            return 0.0, 0
        return (np.sum(df.is_compact) / df.shape[0]) * 100, df.shape[0]

    file_or_dir = Path(file_or_dir)

    if not file_or_dir.exists():
        raise FileNotFoundError(f"{file_or_dir} not found")
    if file_or_dir.is_file():
        return _getpc_and_len(file_or_dir)

    elif file_or_dir.is_dir():
        files = file_or_dir.glob(dir_glob)
        standard_columns = [
            "filename",
            "similarity_metric",
            "similarity_metric_higher_is_better",
            "percent_compact",
            "replicate_criteria",
            "replicate_criteria_not",
            "replicate_query_or_df",
            "null_criteria",
            "null_criteria_not",
            "null_query_or_df",
            "n_contributing",
        ]

        series_list = []
        for file in files:
            if get_percent_compact:
                calculated_pc, len_df = _getpc_and_len(file)
            else:
                calculated_pc, len_df = (None, None)
            d = json.load(open(file.with_suffix(".json")))
            columns = standard_columns.copy()
            new_data = [
                str(file),
                d["similarity_metric_name"],
                d["similarity_metric_higher_is_better"],
                calculated_pc,
                d["replicate_criteria"],
                d["replicate_criteria_not"],
                d["replicate_query"],
                d["null_criteria"],
                d["null_criteria_not"] if d["null_criteria_not"] != [] else np.nan,
                d["null_query_or_df"],
                len_df,
            ]

            additional_data = d.get("additional_captured_params", None)
            if isinstance(additional_data, dict):
                for k, v in additional_data.items():
                    columns.append(k)
                    new_data.append(v)
            series_list.append(pd.Series(data=new_data, index=columns))

        if len(series_list) < 2:
            return pd.DataFrame(series_list).T
        return (
            pd.concat(series_list, axis=1)
            .T.set_index("filename")
            .sort_values("percent_compact", ascending=False)
            .dropna(axis="columns")
        )


def percent_replicating_results_dataframe_to_95pct_confidence_interval(
    df: pd.DataFrame,
    percentile_cutoff: Union[int, float, list[Union[int, float]]] = 95,
    n_resamples: int = 1000,
    similarity_metric_higher_is_better: bool = True,
    n_jobs: int = -1,
) -> Union[list[tuple[float, float]], tuple[float, float]]:
    """Get confidence interval at given percentile cutoff for percent replicating results

    Reads a DataFrame from phenonaut.metrics.performance.percent_replicating
    and performs bootstrapping, sampling from the null distribution to assign a
    confidence interval at the given cutoff, or list of cutoffs.  Returns a tuple
    containing upper and lower 95 % confidence interval bounds.  If multiple
    percentile cutoffs are supplied, then a list containing tuples for each is returned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame supplied by phenonaut.metrics.performance.percent_replicating
    percentile_cutoff:Union[int, float, list[Union[int, float]]]
        Percentile cutoff at which to calculate the confidence interval.  Can
        also be a list, which results in a list of high and low confidence
        interval tuples being returned. Should be between 0 and 100, with a value
        of 95 denoting the 95th percentile as a cutoff. By default 95.
    n_resamples:int
        Number of times to resample the null distribution.  By default 1000.
    similarity_metric_higher_is_better: bool
        If True, then consider treatment replicating if score is greater than
        the percentile cutoff.  If False, then consider treatment replicating if
        score is less than percentile cutoff. By default True.
    n_jobs argument is passed to joblib for parallel execution and defines
        number of threads to use.  A value of -1 denotes that the system should
        determine how many jobs to run. By default -1.
    Returns
    -------
    Union[list[tuple[float, float]], tuple[float, float]]
        Tuple containing 2 values, the first being the lower confidence interval
        and the second being the higher confidence interval.  If multiple
        percentile cutoffs are given, then a list of tuples at each percentile
        cutoff will be returned.
    """
    num_in_null = len(
        [
            1
            for nc in df.columns
            if nc.startswith("null_") and not nc.endswith("_percentile")
        ]
    )

    if df.shape[0] == 0:
        return (
            [None for _ in range(len(percentile_cutoff))]
            if isinstance(percentile_cutoff, (np.ndarray, list, tuple))
            else None
        )

    def _assign_pr(r, p, similarity_metric_higher_is_better):
        if isinstance(r[-num_in_null], str):
            return 0
        if similarity_metric_higher_is_better:
            if r["median_replicate_score"] > np.percentile(r[-num_in_null:], p):
                return 1
            else:
                return 0
        else:
            if r["median_replicate_score"] < np.percentile(r[-num_in_null:], 100 - p):
                return 1
            else:
                return 0

    def _get_pr(df, p, similarity_metric_higher_is_better):
        if df.shape[0] == 0:
            return np.nan
        return (
            np.sum(
                df.apply(
                    _assign_pr,
                    p=p,
                    axis=1,
                    similarity_metric_higher_is_better=similarity_metric_higher_is_better,
                )
            )
            / df.shape[0]
        )

    def _get_null(row):
        if isinstance(row[-num_in_null], str):
            print(row[-num_in_null])
            return np.full(num_in_null, np.nan)
        else:
            return np.array(row[-num_in_null:])

    if isinstance(percentile_cutoff, (float, int)):
        percentile_cutoff = [percentile_cutoff]
    bootstrap_results = []
    null_distributions = np.apply_along_axis(_get_null, 1, df.values)

    def _pfunc_higher_is_better(percentile_cutoff):
        return (
            np.sum(
                repeat_medians
                > np.percentile(
                    np.apply_along_axis(
                        lambda x: np.random.choice(x, num_in_null, replace=True),
                        1,
                        null_distributions,
                    ),
                    percentile_cutoff,
                    axis=1,
                )
            )
            / repeat_medians.shape[0]
        ) * 100

    def _pfunc_higher_is_not_better(percentile_cutoff):
        ndist = np.apply_along_axis(
            lambda x: np.random.choice(x, num_in_null, replace=True),
            1,
            null_distributions,
        )
        return (
            np.sum(
                repeat_medians < np.percentile(ndist, 100 - percentile_cutoff, axis=1)
            )
            / repeat_medians.shape[0]
        ) * 100

    repeat_medians = df.median_replicate_score.values
    for p in percentile_cutoff:
        # repeat_pr=np.empty(n_resamples)
        if similarity_metric_higher_is_better:
            repeat_pr = Parallel(n_jobs=n_jobs)(
                delayed(_pfunc_higher_is_better)(p) for _ in range(n_resamples)
            )

        else:
            repeat_pr = Parallel(n_jobs=-1)(
                delayed(_pfunc_higher_is_not_better)(p) for _ in range(n_resamples)
            )

        bootstrap_results.append(
            (
                np.mean(repeat_pr),
                np.percentile(repeat_pr, 2.5),
                np.percentile(repeat_pr, 97.5),
            )
        )
    if len(bootstrap_results) == 1:
        return bootstrap_results[0]
    else:
        return bootstrap_results
