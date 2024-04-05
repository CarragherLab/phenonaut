# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from collections.abc import Callable
from copy import deepcopy
from optparse import Option
from pathlib import Path
from typing import Optional, Union

import numpy as np
import optuna
import pandas as pd

import phenonaut
import phenonaut.data
from phenonaut.data.dataset import Dataset
from phenonaut.predict.predict_utils import (
    PredictionType,
    get_best_predictor_dataset_df,
    get_df_from_optuna_db,
    get_metric,
    get_prediction_type_from_y,
    get_X_y,
    get_y_from_target,
)
from phenonaut.predict.predictor_dataclasses import (
    PhenonautPredictionMetric,
    PhenonautPredictor,
)
from phenonaut.utils import check_path

from .optuna_functions import run_optuna_opt, run_optuna_opt_merge_folds


def profile(
    phe: phenonaut.Phenonaut,
    output_directory: str,
    dataset_combinations: Optional[Union[None, list[int], list[str]]] = None,
    target: Optional[Union[str, pd.Series, np.ndarray, phenonaut.data.Dataset]] = None,
    predictors: Optional[list[PhenonautPredictor]] = None,
    prediction_type: Optional[Union[str, PredictionType]] = None,
    n_splits: int = 5,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    optuna_db_path: Optional[Union[Path, str]] = None,
    optuna_db_protocol: str = "sqlite:///",
    n_optuna_trials=20,
    metric: Optional[Union[PhenonautPredictionMetric, str]] = None,
    no_output: bool = False,
    write_pptx: bool = True,
    optuna_merge_folds: bool = False,
    test_set_fraction: float = 0.2,
):
    """Profile predictors in their ability to predict a given target.

    This predict.profile function operates on a Phenonaut object (optionally
    containing multiple Datasets) and a given or indicated prediction target.
    The data within the prediction target is examined and the prediction type
    determined from classification, regression, and multiregression/view
    prediction (prediction of 1 omics view from another). In the case of
    Example 1 in the Phenonaut paper, on TCGA, with the prediction target of
    “survives_1_year”, the data types within the metadata are examined and only
    two values found 0 (no) or 1 (yes). Classification is chosen. With no
    classifiers explicitly given as arguments to the profile function,
    Phenonaut selects all default classifiers. User supplied classifiers and
    predictors may be passed, including PyTorch neural networks and similar
    objects through wrapping in PhenonautPredictor dataclasses of any class that
    implements the fit and predict methods. See PhenonautPredictor API
    documentation for further information on including user defined and packaged
    predictors. With no views explicitly listed, all possible view combinations
    are selected. For TCGA, the four omics views allow 15 view combinations
    (4x singular, 6x double, 4x triple and 1x quad). For each unique view
    combination and predictor, perform the following:


    * Merge views and remove samples which do not have features across currently needed views.

    * Shuffle the samples.

    * Withhold 20% of the data as a test set, to be tested against the trained and hyperparameter optimised predictor.

    * Split the data using 5-fold cross validation into train and validation sets.

    * For each fold, perform Optuna hyperparameter optimisation for the given predictor using the train sets, using hyperparameters described by the default predictors for classification, regression and multiregression.


    Parameters
    ----------
    phe : phenonaut.Phenonaut
        A Phenonaut object containing Datasets for prediction on
    output_directory : str
        Directory into which profiling output (boxplots, heatmaps, CSV, JSON
        and PPTX should be written).
    dataset_combinations : Optional[Union[None, list[int], list[str]]], optional
        If the Phenonaut object contains multiple datasets, then tuples of
        'views' may be specified for exploration. If None, then all combinations
        of available views/Datasets are enumerated and used. By default None.
    target : Optional[Union[str, pd.Series, np.ndarray, phenonaut.data.Dataset]], optional
        The prediction target. May be an array-like structure for prediction
        from aligned views, a string denoting the column in which to find the
        prediction target data within one of the Phenonaut Datasets, or a
        pd.Series, by default None.
    predictors : Optional[list[PhenonautPredictor]], optional
        A list of PhenonautPredictors may be supplied to the function. If None,
        the all suitable predictors for the type of prediction problem that it
        is are selected (through loading of
        default_classifiers, default_regressors, or default_multiregressors from
        phenonaut/predict/default_predictors/ .
        By default None.,
    prediction_type : Optional[Union[str, PredictionType]], optional
        The type of prediction task like "classification", "regression" and
        "view", or the proper PredictionType enum. If None, then the prediction
        task is assigned through inspection of the target data types and values
        present.  By default, None.
    n_splits : int, optional
        Number of splits to use in the N-fold cross validation, by default 5.
    random_state : Union[int, np.random.Generator], optional
        If an int, then use this to seed a np.random.Generator for reproducibility of
        random operations (like shuffling etc). If a numpy.random.Generator, then this
        is used as the source of randomness. Can also be None, in which case a random
        seed is used. By default, None.
    optuna_db_path : Optional[Union[Path, str]], optional
        Path to Optuna sqlite3 database file. If None, then a default filename
        will be assigned by Phenonaut. By default None.
    optuna_db_protocol : _type_, optional
        Protocol that Optuna should use for accessing its required persistent
        storage. By default "sqlite:///"
    n_optuna_trials : int, optional
        Number of Optuna trials for hyperparameter optimisation, by default 20
    metric : Optional[Union[PhenonautPredictionMetric, str]], optional
        Metric used for scoring, currently understands the shortcut strings:
        accuracy, accuracy_score
        mse, MSE, mean_squared_error
        rmse, RMSE, root_mean_squared_error
        AUROC, auroc, area_under_roc_curveby default None
    no_output : bool, optional
        If True, then no output is writen to disk. Hyperparameter optimisation
        is performed and the (usually) sqlite3 file written, without writing
        boxplot and heatmap images, CSVs, JSONS, and PPTX files
        By default False.
    write_pptx : bool, optional
        If True, then the output boxplots and heatmaps are written to a PPTX
        file for presentation/sharing of data. By default True.
    optuna_merge_folds : bool, optional
        By default, each fold has hyperparameters optimised and the trained
        predictor with parameters reported.  If this optuna_merge_folds is true,
        then each fold is trained on and and hyperparameters optimised across
        folds (not per-fold). Setting this to False may be useful depending on
        the intended use of the predictor. It is believed that when False, and
        parameters are not optimised across folds, then more accurate prediction
        variance/accuracy estimates are produced. By default False.
    test_set_fraction : float, optional
        When optimising a predictor, by default a fraction of the total data is
        held back for testing, separate from the train-validation splits. This
        test_set_fraction controls the size of this split. By default 0.2.
    save_models : bool, optional
        Save the trained models to pickle files for later use. By default False.

    """
    output_directory = check_path(output_directory, is_dir=True)

    if isinstance(random_state, int):
        np_rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
    else:
        if random_state is None:
            np_rng = np.random.default_rng(None)
        else:
            raise ValueError(
                f"random_state must be an int, numpy.random.Generator, or None. It was {type(random_state)}"
            )

    # Set up optuna_db_path. Dont do any file checks on it - as a sql
    # database may be specified in combination with the optuna_db_protocol
    # argument.
    if optuna_db_path is None:
        optuna_db_path = output_directory / "optuna_scan.db"
    if (
        predictors == []
        and optuna_db_protocol == "sqlite:///"
        and not optuna_db_path.exists()
    ):
        raise ValueError(
            "predictors argument was empty, no optuna_db_path given and no 'optuna_scan.db' was found in the targeted output directory"
        )

    if dataset_combinations is None:
        dataset_combinations = phe.get_dataset_combinations()

    # Turn dataset_combination int tuples into named sets (may happen if user supplied)
    #    ((0,), (0,1))
    dataset_combinations = [
        [ds if isinstance(ds, str) else phe[ds].name for ds in comb]
        for comb in dataset_combinations
    ]

    # Set prediction_type - may be supplied as a string, matching one of the defined enum types.
    # If string, then assign the proper enum type. If None, then detect it.
    if isinstance(prediction_type, str):
        if prediction_type not in [pt.name for pt in PredictionType]:
            raise ValueError(
                f"prediction_type supplied as a string was not a valid option. It was '{prediction_type}', and should be one of:{'.'.join([pt.name for pt in PredictionType])}"
            )
        prediction_type = PredictionType[prediction_type]

    if prediction_type is None:
        # We need to get the prediction target - even before iterating over
        # datasets so that the prediction type can be set below.
        y = get_y_from_target(phe.datasets, target=target)
        prediction_type = get_prediction_type_from_y(y)
        print("Predicted prediction type = ", prediction_type.name)
        if metric is None:
            if prediction_type == PredictionType.classification:
                print("Setting metric to accuracy")
                metric = "accuracy"
            else:
                print("Setting metric to MSE")
                metric = "mse"
    metric = get_metric(metric)

    if n_optuna_trials > 0:
        # If not supplied, set the default predictors
        if predictors is None:
            if prediction_type == PredictionType.classification:
                from phenonaut.predict.default_predictors.classifiers import (
                    default_classifiers,
                )

                predictors = default_classifiers
            if prediction_type == PredictionType.regression:
                from .default_predictors.regressors import default_regressors

                predictors = default_regressors
            if prediction_type == PredictionType.view:
                from .default_predictors.multiregressors import default_multiregressors

                predictors = default_multiregressors

        for dataset_combination in dataset_combinations:
            print(f"Working on datasets: {dataset_combination=}")
            for predictor in predictors:
                X, y = get_X_y(
                    phe, dataset_combination, target, predictor, prediction_type
                )

                if X is None or y is None:
                    continue
                if (
                    prediction_type == PredictionType.classification
                    and predictor.max_classes is not None
                    and predictor.max_classes < len(np.unique(y))
                ):
                    continue
                if (
                    predictor.dataset_size_cutoff is not None
                    and predictor.dataset_size_cutoff < len(y)
                ):
                    continue

                # Split out a test set, leaving train and validation in X and y
                shuffled_indexes = np.arange(y.shape[0], dtype=int)
                np_rng.shuffle(shuffled_indexes)

                if isinstance(X, list):
                    X_test = [
                        Xv[shuffled_indexes[: int(y.shape[0] * test_set_fraction)]]
                        for Xv in X
                    ]
                    X = [
                        Xv[shuffled_indexes[int(y.shape[0] * test_set_fraction) :]]
                        for Xv in X
                    ]
                else:
                    X_test = X[shuffled_indexes[: int(y.shape[0] * test_set_fraction)]]
                    X = X[shuffled_indexes[int(y.shape[0] * test_set_fraction) :]]
                y_test = y[shuffled_indexes[: int(y.shape[0] * test_set_fraction)]]
                y = y[shuffled_indexes[int(y.shape[0] * test_set_fraction) :]]

                if isinstance(target, Dataset):
                    target_dataset_name = target.name
                else:
                    target_dataset_name = None

                if optuna_merge_folds:
                    run_optuna_opt_merge_folds(
                        X,
                        X_test,
                        y,
                        y_test,
                        prediction_type=prediction_type,
                        predictor=predictor,
                        metric=metric,
                        n_optuna_trials=n_optuna_trials,
                        phe_name=phe.name,
                        dataset_combination=dataset_combination,
                        optuna_db_path=optuna_db_path,
                        n_splits=n_splits,
                        random_state=random_state,
                        optuna_db_protocol=optuna_db_protocol,
                        target_dataset_name=target_dataset_name,
                    )
                else:
                    run_optuna_opt(
                        X,
                        X_test,
                        y,
                        y_test,
                        prediction_type=prediction_type,
                        predictor=predictor,
                        metric=metric,
                        n_optuna_trials=n_optuna_trials,
                        phe_name=phe.name,
                        dataset_combination=dataset_combination,
                        optuna_db_path=optuna_db_path,
                        n_splits=n_splits,
                        random_state=random_state,
                        optuna_db_protocol=optuna_db_protocol,
                        target_dataset_name=target_dataset_name,
                    )

    if no_output:
        return

    # Write all scan results to CSV and JSON
    df_all = get_df_from_optuna_db(
        output_directory / "optuna_scan.db",
        csv_output_filename=output_directory / "optuna_scan_all.csv",
        json_output_filename=output_directory / "optuna_scan_all.json",
    )

    # Write best results to CSV and JSON
    df_best = get_df_from_optuna_db(
        output_directory / "optuna_scan.db",
        csv_output_filename=output_directory / "optuna_scan_best.csv",
        json_output_filename=output_directory / "optuna_scan_best.json",
        get_only_best_per_study=True,
    )

    # At this stage, we have written out CSV/JSON files.  Visualisations are sensitive to extreme values,
    # in the case of the GaussianProcess Regressor, it performs teribly in predicting Iris stem length, with an
    # RMSE of ~700 compared to ~0.1 for the rest of the predictors. We may therefore consider removing predictors
    # from visualisations which have score a certain number of StdDev from mean.
    # Experimentation focused on removing items 5x standard deviations away from the mean, however, this is
    # currently left out and remains unimplemented and untested.
    # df_all=df_all[np.abs(np.array((df_all.score-df_all.score.mean())) <= 5*(df_all.score.quantile(0.25)-df_all.score.quantile(0.75)))]
    # df_best=df_best[np.abs(np.array((df_best.score-df_best.score.mean())) <= 5*(df_best.score.quantile(0.25)-df_best.score.quantile(0.75)))]
    if optuna_merge_folds:
        best_with_folds_as_scores = []
        df_best = df_best.reset_index()  # make sure indexes pair with number of rows

        # When plotting the best predictors, we want boxplots to contain the fold results.
        # Therefore we create a new dataframe containing KFold instances of FoldScore,
        # which will be used to plot the results.
        for index, row in df_best.iterrows():
            for fold in row[df_best.columns.str.startswith("FoldScore_")].values:
                tmp_dict = row.to_dict()
                tmp_dict["FoldScore"] = fold
                best_with_folds_as_scores.append(tmp_dict)
        df_best = pd.DataFrame(best_with_folds_as_scores)
        best_predictor_dataset_heatmap = get_best_predictor_dataset_df(df_best)
        best_predictor_std_dataset_heatmap = get_best_predictor_dataset_df(
            df_best, column_containing_values="Std_KFold"
        )
        from phenonaut.output.heatmap import write_heatmap_from_df

        write_heatmap_from_df(
            best_predictor_dataset_heatmap.dropna(how="all"),
            "",
            output_file=output_directory / "best_predictor_ds_heatmap.png",
            standard_deviations_df=best_predictor_std_dataset_heatmap.dropna(how="all"),
            axis_labels=("Predictors", "Views"),
            transpose=False,
            lower_is_better=metric.lower_is_better,
        )

        from phenonaut.output.boxplot import write_boxplot_to_file

        boxplot_best_paths = []
        for dataset in df_best.dataset.unique():
            boxplot_best_paths.append(
                output_directory / f"boxplot_{dataset.replace(':', '_')}_best.png"
            )
            write_boxplot_to_file(
                df_best.query(f"dataset == '{dataset}'"),
                "predictor_name",
                "FoldScore",
                boxplot_best_paths[-1],
                f"{dataset}, best trials, all folds",
                "Predictor",
                df_best.query(f"dataset == '{dataset}'")["metric_name"].unique()[0],
            )

    else:
        best_with_folds_as_scores = []
        df_best = df_best.reset_index()  # make sure indexes pair with number of rows
        mean_df = (
            df_best.groupby(["dataset", "predictor_name"])["test_score"]
            .mean()
            .reset_index()[["dataset", "predictor_name", "test_score"]]
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
        )
        sd_df = (
            df_best.groupby(["dataset", "predictor_name"])["test_score"]
            .std()
            .reset_index()[["dataset", "predictor_name", "test_score"]]
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
        )

        # When plotting the best predictors, we want boxplots to contain the fold results.
        # Therefore we create a new dataframe containing KFold instances of FoldScore,
        # which will be used to plot the results.
        # for index, row in mean_df.iterrows():
        #     tmp_dict = row.to_dict()
        #     tmp_dict["FoldScore"] = fold
        #     best_with_folds_as_scores.append(tmp_dict)
        # df_best = pd.DataFrame(best_with_folds_as_scores)
        # df_best.to_csv(output_directory/"b.csv")
        best_predictor_dataset_heatmap = get_best_predictor_dataset_df(
            mean_df, column_containing_values="test_score"
        )
        best_predictor_std_dataset_heatmap = get_best_predictor_dataset_df(
            sd_df, column_containing_values="test_score"
        )

        best_predictor_dataset_heatmap = best_predictor_dataset_heatmap.reindex(
            df_best.predictor_name.unique(), axis=1
        ).reindex(df_best.dataset.unique(), axis=0)
        best_predictor_std_dataset_heatmap = best_predictor_std_dataset_heatmap.reindex(
            df_best.predictor_name.unique(), axis=1
        ).reindex(df_best.dataset.unique(), axis=0)

        from phenonaut.output.heatmap import write_heatmap_from_df

        write_heatmap_from_df(
            best_predictor_dataset_heatmap.dropna(how="all"),
            "",
            output_file=output_directory / "best_predictor_ds_heatmap.png",
            standard_deviations_df=best_predictor_std_dataset_heatmap.dropna(how="all"),
            axis_labels=("Predictors", "Views"),
            transpose=False,
            lower_is_better=metric.lower_is_better,
        )

        from phenonaut.output.boxplot import write_boxplot_to_file

        boxplot_best_paths = []
        for dataset in df_best.dataset.unique():
            boxplot_best_paths.append(
                output_directory / f"boxplot_{dataset.replace(':', '_')}_best.png"
            )
            write_boxplot_to_file(
                df_best.query(f"dataset == '{dataset}'"),
                "predictor_name",
                "test_score",
                boxplot_best_paths[-1],
                f"{dataset}, best trials, all folds",
                "Predictor",
                df_best.query(f"dataset == '{dataset}'")["metric_name"].unique()[0],
            )
    if write_pptx:
        from phenonaut.output.pptx import PhenonautPPTX

        ppt = PhenonautPPTX(cover_subtitle=phe.name)
        # Make dataset name from the path (2 options here, ends with _best or _all)
        dfromp = (
            lambda f: str(f.stem).replace("_best", "").replace("boxplot_", "")
            if str(f.stem).endswith("_best")
            else str(f.stem).replace("_all", "").replace("boxplot_", "")
        )

        # Sort filenames first by alphabetical, then by length then iterate
        # Unneded: for f in sorted(boxplot_image_filenames, key=lambda x: (len(x.name), x.name)):
        ppt.add_image_slide(
            "Heatmap",
            output_directory / "best_predictor_ds_heatmap.png",
            width=15,
            height=15,
        )
        for f in boxplot_best_paths:
            ppt.add_image_slide(f"{dfromp(f)}", f)
        ppt.save(output_directory / f"prediction_performance.pptx")
