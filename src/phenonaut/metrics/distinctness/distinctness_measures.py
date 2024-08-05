from collections.abc import Callable
from typing import Optional, Union, List
from scipy.spatial.distance import mahalanobis
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from scipy import linalg
from phenonaut import Phenonaut
from ...errors import NotEnoughRowsError
from phenonaut.data import Dataset
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
from sklearn.decomposition import PCA


def silhouette_score(
    ds: Union[Dataset, Phenonaut, pd.DataFrame],
    perturbation_column: Union[str, None],
    replicate_criteria: Optional[Union[str, list[str]]] = None,
    features: Optional[list[str]] = None,
    similarity_metric: Union[str, Callable] = "euclidean",
    similarity_metric_higher_is_better: bool = True,
    return_full_performance_df: bool = False,
):
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
                    "Warning, special similarity metric value"
                    f" {similarity_metric} passed with"
                    " similarity_metric_higher_is_better ="
                    f" {similarity_metric_higher_is_better}, changing"
                    " similarity_metric_higher_is_better to"
                    f" {special_similarity_metrics_higher_better_lookup[similarity_metric]}"
                )
                similarity_metric_higher_is_better = (
                    not similarity_metric_higher_is_better
                )

    if isinstance(similarity_metric, PhenotypicMetric):
        similarity_metric_higher_is_better = similarity_metric.higher_is_better
        if similarity_metric.is_magic_string:
            similarity_metric = similarity_metric.func

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    if isinstance(ds, Dataset):
        df = ds.df
        features = ds.features

        if perturbation_column is None:
            if ds.perturbation_column is None:
                raise ValueError(
                    "No pertuabtion column set, and no perturbation_column argument"
                    " given"
                )
            perturbation_column = ds.perturbation_column
    else:  # Must be df-like
        if perturbation_column is None:
            raise ValueError(
                "Must provide a perturbation_column argument when supplying a DataFrame"
            )
        if features is None:
            raise ValueError(
                "Must provide features argument when supplying a DataFrame"
            )
        df = df

    group_by = [perturbation_column]
    if replicate_criteria is not None:
        if isinstance(replicate_criteria, str):
            replicate_criteria = [replicate_criteria]
        group_by.extend(replicate_criteria)

    gb = df.groupby(group_by)
    n_clusters = len(gb)

    scores = np.eye(n_clusters)
    scores[scores == 1] = np.nan

    for i, (group_name_a, indices_a) in enumerate(gb.indices.items()):
        for j, (group_name_b, indices_b) in enumerate(gb.indices.items()):
            if group_name_a == group_name_b:
                continue
            X = np.vstack([df.iloc[indices_a][features], df.iloc[indices_b][features]])
            labels = [
                *([group_name_a] * len(indices_a)),
                *([group_name_b] * len(indices_b)),
            ]
            scores[i, j] = sklearn_silhouette_score(X, labels, metric=similarity_metric)
    scores_df = pd.DataFrame(
        data=scores, index=list(gb.indices.keys()), columns=list(gb.indices.keys())
    )
    scores_df["mean_silhouette_score"] = np.nanmean(scores, axis=1)
    if return_full_performance_df:
        return scores_df["mean_silhouette_score"].mean(), scores_df
    else:
        return scores_df["mean_silhouette_score"].mean()


def mp_value_score(
    ds: Union[Dataset, Phenonaut],
    ds_groupby: Union[str, List[str]],
    reference_perturbation_query: str,
    pca_explained_variance: float = 0.99,
    std_scaler_columns: bool = True,
    std_scaler_rows: bool = False,
    n_iters: int = 1000,
    random_state: int = 42,
    raise_error_for_low_count_groups: bool = True,
):
    """Get mp-value score performance DataFrame for a dataset

    Implementation of the mp-value score from the paper:
        Hutz JE, Nelson T, Wu H, et al. The Multidimensional Perturbation Value: A
        Single Metric to Measure Similarity and Activity of Treatments in
        High-Throughput Multidimensional Screens. Journal of Biomolecular Screening.
        2013;18(4):367-377. doi:10.1177/1087057112469257.

    The paper mentions normalising by rows as well as columns. This is not appropriate
    for some data types like DRUG-seq, and so this is not enabled by default.
    Additionally, a default fraction explained variance for the PCA operation has been
    set to 0.99 so that the PCA may explain 99 % of variance.

    This implementation differs somewhat to the one in pycytominer_eval which deviates
    from the paper definition and does not perform a mixin of the covariance matrices
    for treatment and control.

    Parameters
    ----------
    ds : Union[Dataset, Phenonaut]
        Phenonaut dataset or Phenonaut object upon which to perform the mp_value_score
        calculation. If a Phenonaut object is passed, then the dataset at position -1
        (usually the last added is used)
    ds_groupby: Pandas style groupby to apply on the ds. Normally this is the column
        name of a unique compound identifier. Can also be a list, containing the unique
        compound identifier column name, along with a concentration or timepoint column.
    reference_perturbation_query : reference_perturbation_query
        Pandas style query which may be run on ds to extract the reference set of points
        in phenotypic space, against which all other grouped perturbations are compared.
    pca_explained_variance: float
        This argument is passed to scikit's PCA object and specifices the % variance
        that the returned components should capture. The original paper aims for 90 % ev
        we aim by default for 99 %. Should be expressed as a float between 0 and 1.
        By default 0.99
    std_scaler_columns: bool
        Apply standard scaler to columns. By default True
    std_scaler_rows: bool
        Apply standard scaler to rows. By default False
    n_iters : int
        Number of iterations iterations to perform in statistical test to derive
        p-value, by default 1000
    n_jobs : int, optional
        Calculations will be run in parallel by providing the number of processors to
        use. If n_jobs is None, then this is autodetected by the system. By default None
    random_state : int
        Random seed to use for initialisation of rng, enabling reproducible runs
    raise_error_for_low_count_groups : bool
        Calculation of mp_value scores requires more than three samples to be in each
        group. If raise_error_for_low_count_groups is True, then an error is raised upon
        encountering such a group as no mp_value score can be calculated. If False, then
        a simple warning is printed and the returned p-value and mahalanobis distance in
        the results dataframe are both np.nan. By default True
    """

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    vehicle_data = ds.df.query(reference_perturbation_query)[ds.features].values
    if len(vehicle_data) < 3:
        raise NotEnoughRowsError(
            f"Vehicle had {len(vehicle_data)} rows. 3 or more are required for the"
            " mp_value_score calculation. It is highly likely that the query string"
            " passed as reference_perturbation_query does not return any dataframe"
            " rows. Try running the supplied query on your dataset.df and see if"
            f" anything is returned.  Query was: {reference_perturbation_query}"
        )
    len_veh_data = vehicle_data.shape[0]
    grouped_df = ds.df.groupby(ds_groupby)

    mp_value_score_results_df = pd.DataFrame()

    for group_identifier, group_df in grouped_df:
        print("Working on", group_identifier, len(group_df))
        if len(group_df) < 3:
            if raise_error_for_low_count_groups:
                raise NotEnoughRowsError(
                    f"Group '{group_identifier}' contained only {len(group_df)} rows. 3"
                    " or more are required for the mp_value_score calculation. To"
                    " continue processing when small groups like this are encountered,"
                    " call mp_value_score again with raise_error_for_low_count_groups"
                    " = False"
                )
            print(
                f"WARNING, Group '{group_identifier}' contained only"
                f" {len(group_df)} rows. 3 or more are required for the mp_value_score"
                " calculation. np.nan values will be included in results. Continuing"
                " as raise_error_for_low_count_groups was False."
            )
            mp_value_score_results_df.loc[
                group_identifier, ["mahalanobis_distance", "mp_value"]
            ] = (np.nan, np.nan)
            continue
        # Reseed RNG for every group otherwise the order of groups would influenc
        # mp_value scores
        np_rng = np.random.default_rng(random_state)

        trt_data = group_df[ds.features].values
        len_trt_data = trt_data.shape[0]
        X = np.vstack([trt_data, vehicle_data])
        # Scale data
        if std_scaler_columns:
            X = (X - X.mean(axis=0)) / np.std(X, axis=0)
        if std_scaler_rows:
            X = (X - X.mean(axis=1)) / np.std(X, axis=1)

        pca = PCA(n_components=pca_explained_variance, svd_solver="full")
        X = pca.fit_transform(X)
        X = X * pca.explained_variance_ratio_

        # Compute covariance matrices of the scaled data and add them before inverting for
        # calculation of mahalanobis distances
        cov_trt = (np.cov(X[:len_trt_data].T)) * (
            len_trt_data / (len_trt_data + len_veh_data)
        )
        cov_veh = (np.cov(X[len_trt_data:].T)) * (
            len_veh_data / (len_trt_data + len_veh_data)
        )
        inv_cov = linalg.inv(cov_trt + cov_veh)

        veh_pca_mean = np.mean(X[len_trt_data:], axis=0)
        original_m_dist = mahalanobis(
            np.mean(X[:len_trt_data], axis=0), veh_pca_mean, inv_cov
        )

        original_labels = np.array([1] * len_trt_data + [0] * len_veh_data)
        cur_labels = original_labels.copy()
        perturbed_distances = np.empty(n_iters)
        for i in range(n_iters):
            np_rng.shuffle(cur_labels)
            while np.all(cur_labels[:len_trt_data]):
                np_rng.shuffle(cur_labels)
            not_cur_labels = np.logical_not(cur_labels)
            pcov_trt = np.cov(X[np.where(cur_labels)].T) * (
                len_trt_data / (len_trt_data + len_veh_data)
            )
            pcov_veh = np.cov(X[np.where(not_cur_labels)].T) * (
                len_veh_data / (len_trt_data + len_veh_data)
            )
            pinv_cov = linalg.inv(pcov_trt + pcov_veh)

            perturbed_distances[i] = mahalanobis(
                np.mean(X[np.where(cur_labels)], axis=0),
                np.mean(X[np.where(not_cur_labels)], axis=0),
                pinv_cov,
            )
            mp_value = np.mean([v > original_m_dist for v in perturbed_distances])
        mp_value_score_results_df.loc[
            group_identifier, ["mahalanobis_distance", "mp_value"]
        ] = (original_m_dist, mp_value)
    return mp_value_score_results_df
