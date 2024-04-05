# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from enum import Enum, auto
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

import phenonaut.data
from phenonaut.data import dataset
from phenonaut.data.dataset import Dataset
from phenonaut.phenonaut import Phenonaut
from phenonaut.predict.predictor_dataclasses import (
    PhenonautPredictionMetric,
    PhenonautPredictor,
)


# The PredictionType enum is used in multiple places to denote the prediction task.
# 'view' is simply multi-y-regression for prediction either different views, or multiple
# properties at the same time
class PredictionType(Enum):
    """PredictionType Enum for classification, regression or view prediction.

    Parameters
    ----------
    Enum : int
        Enumerated type, captures if the prediction task is classification,
        regression, or view (multiregression)
    """

    classification = auto()
    regression = auto()
    view = auto()


def get_y_from_target(
    data: Union[
        phenonaut.data.Dataset,
        pd.DataFrame,
        list[phenonaut.data.Dataset],
        list[pd.DataFrame],
    ],
    target: Optional[Union[str, pd.Series, np.ndarray, tuple]] = None,
) -> Union[list, pd.Series, tuple, np.ndarray]:
    """Get target y from Dataset(s) or DataFrame

    Parameters
    ----------
    data : Union[list[Dataset],list[DataFrame]]
        Dataset, DataFrame, list of Datasets or list of DataFrames where the
        target values are present.
    target : Optional[Union[str, Series, np.ndarray, tuple]], optional
        Target column name containing target y values in Dataset/DataFrame. If
        None, then the target y is attempted to be inferred by looking for
        Dataset columns which are not listed as features. Target y cannot be
        inferred from DataFrames. If np.ndarray, tuple, or pd.Series, then this
        is directly returned as the target of prediciton y, by default None.

    Returns
    -------
    Union[list, Series, tuple, np.ndarray]
        Target y values for prediction

    Raises
    ------
    ValueError
        Given target string (column title) was a feature of the dataset.
    ValueError
        Object should be phenonaut.Dataset or pd.DataFrame.
    ValueError
        Given target string (column title) was not found in any supplied
        Datasets/Dataframes.
    ValueError
        Target was not set, trying to guess it from a phenonaut.Dataset, but
        did not find this type.
    ValueError
        Could not guess the target from supplied Dataset(s). Pass target as
        string for a dataset.df column heading, or the prediction target
        directly.
    """
    # If target is directly given (series, ndarray, tuple, list), then return it

    if isinstance(target, (pd.Series, np.ndarray, tuple, list)):
        return target
    if isinstance(target, Dataset):
        return target.data

    # Processing bellow expects data to be in a list Data should always be a list
    if not isinstance(data, list):
        data = [data]

    # If target is given, use it to set y.
    if target is not None:
        # If string, then search through the datasets to find a column with that value, use the first found
        if isinstance(target, str):
            for d in data:
                if isinstance(d, pd.DataFrame):
                    if target in d.columns.values:
                        return d[target]
                elif isinstance(d, phenonaut.data.Dataset):
                    if target in d.features:
                        raise (
                            f"Given target string (column title) was {target}, but this is a feature of the dataset (name={d.name}, features={d.features}"
                        )
                    if target in d.get_non_feature_columns():
                        return d.df[target]
                else:
                    raise ValueError(
                        f"Object of type {type(d)} found, should be phenonaut.Dataset or pd.DataFrame"
                    )
            raise ValueError(
                f"Given target string (column title) was {target}, but this was not found in any supplied Datasets/Dataframes"
            )

    # Target was not set, guess it by identifying a ds that has one column not listed as a feature.
    for d in data:
        if not isinstance(d, phenonaut.data.Dataset):
            raise ValueError(
                f"Target was not set, trying to guess it from a phenonaut.Dataset, but found {type(d)}"
            )
        if "prediction_target" in d._metadata.keys():
            return d.df[d._metadata["prediction_target"]]
        non_feature_columns = d.get_non_feature_columns()
        if len(non_feature_columns) == 1:
            return d.df[non_feature_columns[0]]
    raise ValueError(
        "Could not guess the target from supplied Dataset(s). Pass target as string for a dataset.df column heading, or the prediction target directly"
    )


def get_prediction_type_from_y(y):
    """For a given target y - get prediction type

    Looking at the data in y, return prediction type from classification,
    regression or multiregression (view prediction).

    Parameters
    ----------
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if isinstance(y, pd.Series):
        if y.values.dtype in [int, np.int64]:
            return PredictionType.classification
        else:
            return PredictionType.regression
    if isinstance(y, list):
        if type(y[0]) == int:
            return PredictionType.classification
        else:
            return PredictionType.regression
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            if y.dtype == int:
                return PredictionType.classification
            else:
                return PredictionType.regression
        else:
            return PredictionType.view
    if isinstance(y, pd.DataFrame):
        if y.values.shape[1] > 1:
            return PredictionType.view
        else:
            if y.iloc[0, 0].values.dtype == int:
                return PredictionType.classification
            else:
                return PredictionType.regression
    if isinstance(y, phenonaut.data.Dataset):
        return PredictionType.view
    raise ValueError(
        f"y was not pd.Series, list, np.array, phenonaut.data.Dataset, it was : {type(y)}"
    )


def get_common_indexes(dataframes_list: list[pd.DataFrame]) -> list[str]:
    """Get common indexes from list of DataFrames

    Parameters
    ----------
    dataframes_list : list[pd.DataFrame]
        List of pd.DataFrames from which common indexes should be extracted.

    Returns
    -------
    list[str]
        List of common indexes between pd.DataFrames.
    """
    return list(reduce(lambda x, y: x & y, [set(df.index) for df in dataframes_list]))


def get_df_from_optuna_db(
    optuna_db_file: Union[str, Path],
    csv_output_filename: Union[Path, str] = None,
    json_output_filename: Union[Path, str] = None,
    get_only_best_per_study: bool = False,
) -> pd.DataFrame:
    """After predict.profile, turn Optuna sqlite3 files into pd.DataFrame

    Parameters
    ----------
    optuna_db_file : Union[str, Path]
        Optuna hyperparameter optimisation database (sqlite3file)
    csv_output_filename : Union[Path, str], optional
        Target output CSV file. Can be None, in which case, no CSV file is
        written out. By default None.
    json_output_filename : Union[Path, str], optional
        Target output JSON file. Can be None, in which case, no JSON file is
        written out. By default None.
    get_only_best_per_study : bool, optional
        Boolean value stating if only the best hyperparameter set per study
        should be writen out. By default False.

    Returns
    -------
    pd.DataFrame
        DataFrame sumarising Optuna hyperparameter scan results.

    Raises
    ------
    FileNotFoundError
        Database file (sqlite3) not found.
    """
    import optuna

    df_dict_list = []
    if isinstance(optuna_db_file, str):
        optuna_db_file = Path(optuna_db_file)
    if not optuna_db_file.exists():
        raise FileNotFoundError(
            f"Could not find optuna database file ({optuna_db_file})"
        )
    study_names = [
        study.study_name
        for study in optuna.study.get_all_study_summaries(
            storage=f"sqlite:///{optuna_db_file.resolve()}"
        )
    ]
    for study_name in study_names:
        study = optuna.study.load_study(
            study_name, storage=f"sqlite:///{optuna_db_file.resolve()}"
        )
        if get_only_best_per_study:
            # If optuna is interuped with CTRL+C when using an sqlite database, then an sqlite database
            # can have a study, but no trials, resulting in a crash here. The following issue addresses
            # the problem:  https://github.com/optuna/optuna/issues/2101
            try:
                trial = study.best_trial
            except ValueError:
                # Optuna/sqlite returned
                # ValueError: Record does not exist.
                # Study does not contain any trials
                continue
            row_dict = {k: v for k, v in study.user_attrs.items()}
            row_dict.update({k: v for k, v in trial.user_attrs.items()})

            if "KFoldScores" in row_dict:
                del row_dict["KFoldScores"]
                fold_scores = trial.user_attrs["KFoldScores"]
                row_dict.update(
                    {
                        f"FoldScore_{i+1}": fold_scores[i]
                        for i in range(len(fold_scores))
                    }
                )
            row_dict.update({"parameters": trial.params})
            df_dict_list.append(row_dict)

        else:  # Getting all
            for trial in study.get_trials():
                row_dict = {
                    k: v
                    for k, v in study.user_attrs.items()
                    if k not in ["best_trained_model_score", "best_trained_model"]
                }
                row_dict.update({k: v for k, v in trial.user_attrs.items()})
                if "KFoldScores" in row_dict:
                    del row_dict["KFoldScores"]
                    fold_scores = trial.user_attrs["KFoldScores"]
                    row_dict.update(
                        {
                            f"FoldScore_{i+1}": fold_scores[i]
                            for i in range(len(fold_scores))
                        }
                    )
                row_dict.update({"parameters": trial.params})
                df_dict_list.append(row_dict)

    df = pd.DataFrame(df_dict_list)
    columns_for_the_end = ["test_score", "parameters"]
    df = df[
        [colname for colname in df if not colname in columns_for_the_end]
        + [endcol for endcol in columns_for_the_end if endcol in df]
    ]
    if csv_output_filename is not None:
        csv_output_filename = Path(csv_output_filename)
        if not csv_output_filename.parent.exists():
            csv_output_filename.mkdir(parents=True)
        df.to_csv(csv_output_filename)
    if json_output_filename is not None:
        json_output_filename = Path(json_output_filename)
        if not json_output_filename.parent.exists():
            json_output_filename.mkdir(parents=True)
        df.to_json(json_output_filename)
    return df


def get_best_predictor_dataset_df(
    df: pd.DataFrame, column_containing_values: str = "test_score"
) -> pd.DataFrame:
    """For a given Optuna hyperaprameter scan dataframe, get the best predictor

    Parameters
    ----------
    df : pd.DataFrame
        Optua hyperparameterscan pd.DataFrame, likely generated by
        get_df_from_optuna_db.
    column_containing_values : str, optional
        Name of the column containing scores. By default "test_score".

    Returns
    -------
    pd.DataFrame
        DataFrame containing information on the best predictor.
    """
    unique_predictors = df["predictor_name"].unique()
    unique_dataset_views = df["dataset"].unique()
    heatmap_data_array = np.full(
        (len(unique_dataset_views), len(unique_predictors)), np.nan
    )

    for pred_i in range(len(unique_predictors)):
        for ds_i in range(len(unique_dataset_views)):
            heatmap_data_array[ds_i, pred_i] = df.query(
                f"predictor_name=='{unique_predictors[pred_i]}' and dataset=='{str(unique_dataset_views[ds_i])}'"
            )[column_containing_values].max()

    return pd.DataFrame(
        heatmap_data_array, index=unique_dataset_views, columns=unique_predictors
    )


def get_metric(metric: Union[str, dict, PhenonautPredictionMetric]):
    """Get metric function from various options for metric definition

    Helper function which allows specification of metrics with strings
    indicating common names, dictionaries.

    Currently understands the shortcut strings:
    accuracy, accuracy_score
    mse, MSE, mean_squared_error
    rmse, RMSE, root_mean_squared_error
    AUROC, auroc, area_under_roc_curve

    Parameters
    ----------
    metric : Union[str, dict, PhenonautPredictionMetric]
        String, dict or PhenonautPredictionMetric to be used for scoring

    Returns
    -------
    PhenonautPredictionMetric
        Prediction metric

    Raises
    ------
    ValueError
        No metrics found matching short string name.
    KeyError
        Given dictionary did not include all required fields.
    ValueError
        metric argument was not of a suitable type.
    """
    if isinstance(metric, PhenonautPredictionMetric):
        return metric
    elif isinstance(metric, str):
        if metric in ["accuracy", "accuracy_score"]:
            from sklearn.metrics import accuracy_score

            return PhenonautPredictionMetric(accuracy_score, "Accuracy", False)

        elif metric in ["mse", "MSE", "mean_squared_error"]:
            from sklearn.metrics import mean_squared_error

            return PhenonautPredictionMetric(mean_squared_error, "MSE", True)

        elif metric in ["rmse", "RMSE", "root_mean_squared_error"]:
            from sklearn.metrics import mean_squared_error

            return PhenonautPredictionMetric(
                lambda x, y: mean_squared_error(x, y, squared=False), "RMSE", True
            )

        elif metric in ["AUROC", "auroc", "area_under_roc_curve"]:
            from sklearn.metrics import roc_auc_score

            return PhenonautPredictionMetric(roc_auc_score, "AUROC", False)

        raise ValueError(
            f"Did not find any predefined metrics matching the given '{metric}' shortcut, try accuracy, mse, rmse, or auroc"
        )

    elif isinstance(metric, dict):
        if not all([k in metric for k in ("func", "name", "lower_is_better")]):
            raise KeyError(
                f"'func', 'name', and 'lower_is_better' should be metric dictionary keys (containing the types Callable, str, bool), but found: {', '.join(metric.keys())}"
            )
        return PhenonautPredictionMetric(
            metric["func"], metric["name"], metric["lower_is_better"]
        )
    else:
        raise ValueError(
            f"Metric should be a str, PhenonautPredictionMetric, or dict, it was: {type(metric)}"
        )


def get_X_y(
    phe: Phenonaut,
    dataset_combination: Union[tuple, list],
    target,
    predictor: PhenonautPredictor,
    prediction_type: PredictionType,
) -> tuple:
    """For a given set of views, and known y, get X and y for predictor training

    Parameters
    ----------
    phe : Phenonaut
        The Phenonaut object containing Datasets
    dataset_combination : Union[tuple, list]
        Dataset combinations to be used in prediction task.
    target : pd.Series, np.ndarray
        Prediction target
    predictor : PhenonautPredictor
        The PhenonautPredictor being used.
    prediction_type : PredictionType
        Enum classification type specifying classification, regression or view.

    Returns
    -------
    tuple
        X, y tuple for training of predictor.
    """
    X = None
    y = get_y_from_target(phe[dataset_combination], target=target)
    common_indices = None
    if isinstance(y, (pd.DataFrame, pd.Series)):
        common_indices = get_common_indexes(
            [phe[dsn].df for dsn in dataset_combination] + [y]
        )
    elif isinstance(y, dataset):
        common_indices = get_common_indexes(
            [phe[dsn].df for dsn in dataset_combination] + [y.data]
        )
    # 1 datasets
    if len(dataset_combination) == 1:
        #  1 view predictor
        if predictor.num_views == 1:
            if prediction_type == PredictionType.view:
                if common_indices is None:
                    return (phe[dataset_combination[0]].data.values, y.values)
                else:
                    return (
                        phe[dataset_combination[0]].data.loc[common_indices, :].values,
                        y.loc[common_indices, :].values,
                    )
            else:
                if common_indices is None:
                    return (phe[dataset_combination[0]].data.values, y.values)
                else:
                    return (
                        phe[dataset_combination[0]].data.loc[common_indices, :].values,
                        y.loc[common_indices].values,
                    )

        # Multiview predictor not compatible with single view data
        else:
            return (None, None)

    # Multiple datasets - multiview
    else:
        if predictor.num_views == 1:  # Predictor can only do 1 view
            if common_indices is None:
                return (
                    np.hstack(
                        [phe[ds_name].data.values for ds_name in dataset_combination]
                    ),
                    y.values,
                )
            else:
                return (
                    np.hstack(
                        [
                            phe[ds_name].data.loc[common_indices, :].values
                            for ds_name in dataset_combination
                        ]
                    ),
                    y.loc[common_indices].values,
                )
        else:  # Predictor is multiview
            if len(dataset_combination) != predictor.num_views:
                return (None, None)
            if common_indices is None:
                return (
                    [phe[ds_name].data.values for ds_name in dataset_combination],
                    y.values,
                )
            else:
                return (
                    [
                        phe[ds_name].data.loc[common_indices, :].values
                        for ds_name in dataset_combination
                    ],
                    y.loc[common_indices].values,
                )
