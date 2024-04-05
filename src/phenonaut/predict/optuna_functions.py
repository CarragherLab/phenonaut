# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import base64
import gzip
from copy import deepcopy
from pathlib import Path
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from timeit import default_timer
from typing import Optional, Union

import numpy as np
import optuna
from sklearn.model_selection import KFold

import phenonaut
from phenonaut.predict.predict_utils import PredictionType
from phenonaut.predict.predictor_dataclasses import (
    PhenonautPredictionMetric,
    PhenonautPredictor,
)


def predictor_to_str(ob: PhenonautPredictor) -> str:
    """Encode a PhenonautPredictor (or any serialisable object) to str

    First serialise the object using pickle.dumps, compress using gzip and
    return the base64 string representation.  Works with any serialisable
    object.

    Parameters
    ----------
    ob : PhenonautPredictor
        Serializable object to convert.

    Returns
    -------
    str
        Base64 gzip serialised object as a utf-8 string.
    """
    return base64.b64encode(gzip.compress(pickle_dumps(ob))).decode("utf-8")


def predictor_from_str(ob_str: str) -> PhenonautPredictor:
    """Convert a base64 gziped object string to PhenonautPredictor

    Convert a str to object after applying gzip decompression and then
    pickle.loads.

    Parameters
    ----------
    ob_str : str
        String representation of base64 encoded gzipped object

    Returns
    -------
    PhenonautPredictor
        PhenonautPredictor object
    """
    return pickle_loads(gzip.decompress(base64.b64decode(ob_str)))


def run_optuna_opt(
    X: Union[list[np.ndarray], np.ndarray],
    X_test: Union[list[np.ndarray], np.ndarray],
    y: Union[list[np.ndarray], np.ndarray],
    y_test: Union[list[np.ndarray], np.ndarray],
    prediction_type: PredictionType,
    predictor: PhenonautPredictor,
    metric: PhenonautPredictionMetric,
    n_optuna_trials: int,
    phe_name: str,
    dataset_combination: list[str],
    optuna_db_path: Union[Path, str],
    n_splits: int = 5,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    optuna_db_protocol: str = "sqlite:///",
    target_dataset_name: Optional[str] = None,
):
    """Run Optuna-led hyperparameter optimisation on predictor and data

    Parameters
    ----------
    X : Union[list[np.ndarray], np.ndarray]
        Training data
    X_test : Union[list[np.ndarray], np.ndarray]
        Test data
    y : Union[list[np.ndarray], np.ndarray]
        Training target
    y_test : Union[list[np.ndarray], np.ndarray]
        Test target data
    prediction_type : PredictionType
        Prediction type specifying classification, regression or view.
    predictor : PhenonautPredictor
        The predictor with fit, predict functions packed into PhenonautPredictor
        dataclass which supports the predictor class with specification of
        hyperparameters etc.
    metric : PhenonautPredictionMetric
        The scoring metric to be used to assess performance.
    n_optuna_trials : int
        Number of optuna trials to optimise across.
    phe_name : str
        Name of the Phenonaut object from which the data comes
    dataset_combination : list[str]
        Combination views of datasets to be used
    optuna_db_path : Union[Path, str]
        Output file path for Optuna sqlite3 database file
    n_splits : int, optional
        Number of splits to be used in cross fold validation, by default 5
    random_state : Optional[Union[int, np.random.Generator]], optional
        Seed for use by random number generator, allow deterministing repeats.
        If None, then do no pre-seed the generator. By default None
    optuna_db_protocol : _type_, optional
        Protocol for optuna to use to access storage. By default "sqlite:///"
    target_dataset_name : Optional[str], optional
        If predicting a view, then the target Dataset may be given a name,
        by default None.
    """

    # First, we make a deepcopy of predictor, as the type is mutable, and we will be changing it in
    # this function, adding a member variable to obtain the trained predictor back from the
    # augmented optuna objective function.
    predictor = deepcopy(predictor)

    if isinstance(random_state, np.random.Generator):
        random_state = random_state.integers(int(1e9))

    # If predictor has max_optuna_trials set lower than max_optuna trials supplied by the function arg,
    # then take the lower. If not set, then we can use all trials denoted by n_optuna_trials argument.
    if predictor.max_optuna_trials is None:
        n_optuna_trials_for_predictor = n_optuna_trials
    else:
        n_optuna_trials_for_predictor = min(
            predictor.max_optuna_trials, n_optuna_trials
        )

    for fold_index, (train_indexes, validation_indexes) in enumerate(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            X[0] if isinstance(X, list) else X
        )
    ):
        if isinstance(X, list):
            X_train = [X[i][train_indexes] for i in range(len(X))]
            X_val = [X[i][validation_indexes] for i in range(len(X))]
        else:
            X_train = X[train_indexes]
            X_val = X[validation_indexes]
        y_train = y[train_indexes]
        y_val = y[validation_indexes]

        # Name the study- must be unique for predictor/dataset/target predictions
        if prediction_type == PredictionType.view and target_dataset_name is not None:
            optuna_study_name = f"{phe_name};{prediction_type.name};{predictor.name};{dataset_combination};{target_dataset_name};fold{fold_index+1}"
        else:
            optuna_study_name = f"{phe_name};{prediction_type.name};{predictor.name};{dataset_combination};fold{fold_index+1}"

        # Load study
        optuna_study = optuna.create_study(
            study_name=optuna_study_name,
            storage=f"{optuna_db_protocol}{optuna_db_path}",
            load_if_exists=True,
            direction="minimize" if metric.lower_is_better else "maximize",
        )
        # Some predictors like GaussianNB do not take any hyperparameters, others are easily explored with
        # some categorical parameters. These therefore require 1 run or very few to explore the entire search
        # space. We therefore see if max_optuna_trials for the predictor has been reached in this study before
        # if so, then there is no reason to test those parameters again.
        if (
            predictor.max_optuna_trials is not None
            and len(optuna_study.trials) >= predictor.max_optuna_trials
        ):
            print(
                f"max_optuna_trials ({predictor.max_optuna_trials}) already reached for study, continuing."
            )
            return

        # It is a new study, set the user attributes for ease of analysis
        if len(optuna_study.trials) == 0:
            optuna_study.set_user_attr("phe_obj_name", phe_name)
            optuna_study.set_user_attr("predictor_name", predictor.name)
            optuna_study.set_user_attr("dataset", ":".join(dataset_combination))
            optuna_study.set_user_attr("prediction_type", prediction_type.name)
            if (
                prediction_type == PredictionType.view
                and target_dataset_name is not None
            ):
                optuna_study.set_user_attr("predicting_view", target_dataset_name)
            optuna_study.set_user_attr("metric_name", metric.name)
            optuna_study.set_user_attr("lower_is_better", metric.lower_is_better)
            optuna_study.set_user_attr("merge_folds", False)
            optuna_study.set_user_attr("fold_index", fold_index + 1)
        start_time = default_timer()
        optuna_study.optimize(
            lambda trial: _optuna_augmented_objective(
                trial, optuna_study, X_train, X_val, y_train, y_val, predictor, metric
            ),
            n_trials=n_optuna_trials_for_predictor,
        )
        optuna_study.set_user_attr("time_taken", default_timer() - start_time)
        test_score = metric(y_test, predictor._best_predictor_instance.predict(X_test))
        optuna_study.set_user_attr("test_score", test_score)


def _optuna_augmented_objective(
    trial: optuna.Trial,
    optuna_study: optuna.Study,
    X_train: Union[list[np.ndarray], np.ndarray],
    X_val: Union[list[np.ndarray], np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    predictor: PhenonautPredictor,
    metric: PhenonautPredictionMetric,
):
    """Flexible Optuna objective

    Parameters
    ----------
    trial : optuna.Trial
        The Optua trial passed by Optuna
    optuna_study : optuna.Study
        Study name
    X_train : Union[list[np.ndarray], np.ndarray]
        Training data
    X_val : Union[list[np.ndarray], np.ndarray]
        Validation data
    y_train : np.ndarray
        Training target data
    y_val : np.ndarray
        Validation target data
    predictor : PhenonautPredictor
        Predictor dataclass holding predictor and possible hyperparameters.
    metric : PhenonautPredictionMetric
        The metric to be used to score predictions.

    Returns
    -------
    float
        Score of trained model.
    """
    optuna_dict = {}
    if predictor.optuna is None:
        optuna_dict = {}
    else:
        optuna_dict = {
            hyperparameter.name: getattr(trial, hyperparameter.optuna_func)(
                hyperparameter.name, *hyperparameter.parameters, **hyperparameter.kwargs
            )
            for hyperparameter in predictor.optuna
        }

    if predictor.conditional_hyperparameter_generator is not None:
        optuna_dict.update(
            predictor.conditional_hyperparameter_generator(
                predictor.conditional_hyperparameter_generator_constructor_keyword,
                trial,
                optuna_dict,
                X_train.shape[1],
                y_train.shape[1] if y_train.ndim == 2 else 1,
            )
        )
    if isinstance(predictor.predictor, tuple):
        predictor_instance = predictor.predictor[0](
            predictor.predictor[1](**optuna_dict, **predictor.constructor_kwargs)
        )
    else:
        predictor_instance = predictor.predictor(
            **optuna_dict, **predictor.constructor_kwargs
        )
    predictor_instance.fit(X_train, y_train)
    score = metric(y_val, predictor_instance.predict(X_val))

    # If not yet set (because its a new study with 0 trials and was initialised to None, then set it to the current score)
    if "best_trained_model_score" not in optuna_study.user_attrs:
        optuna_study.set_user_attr("best_trained_model_score", score)
    if not hasattr(predictor, "_best_predictor_instance"):
        predictor._best_predictor_instance = predictor_instance

    cur_best_score = optuna_study.user_attrs["best_trained_model_score"]

    if (metric.lower_is_better and score <= cur_best_score) or (
        not metric.lower_is_better and score >= cur_best_score
    ):
        optuna_study.set_user_attr("best_trained_model_score", score)
        predictor._best_predictor_instance = predictor_instance

    return score


def run_optuna_opt_merge_folds(
    X: Union[list[np.ndarray], np.ndarray],
    X_test: Union[list[np.ndarray], np.ndarray],
    y: np.ndarray,
    y_test: np.ndarray,
    prediction_type: PredictionType,
    predictor: PhenonautPredictor,
    metric: PhenonautPredictionMetric,
    n_optuna_trials: int,
    phe_name: str,
    dataset_combination: list[str],
    optuna_db_path: Union[Path, str],
    n_splits: int = 5,
    random_state: Optional[int] = None,
    optuna_db_protocol: str = "sqlite:///",
    target_dataset_name: Optional[str] = None,
):
    # First, we make a deepcopy of predictor, as the type is mutable, and we will be changing it in
    # this function, adding a member variable to obtain the trained predictor back from the
    # augmented optuna objective function.
    predictor = deepcopy(predictor)
    # If predictor has max_optuna_trials set lower than max_optuna trials supplied by the function arg,
    # then take the lower. If not set, then we can use all trials denoted by n_optuna_trials argument.
    if predictor.max_optuna_trials is None:
        n_optuna_trials_for_predictor = n_optuna_trials
    else:
        n_optuna_trials_for_predictor = min(
            predictor.max_optuna_trials, n_optuna_trials
        )

    # Name the study- must be unique for predictor/dataset/target predictions
    if prediction_type == PredictionType.view and target_dataset_name is not None:
        optuna_study_name = f"{phe_name};{prediction_type.name};{predictor.name};{dataset_combination};{target_dataset_name};merge_folds"
    else:
        optuna_study_name = f"{phe_name};{prediction_type.name};{predictor.name};{dataset_combination};merge_folds"

    # Load study
    optuna_study = optuna.create_study(
        study_name=optuna_study_name,
        storage=f"{optuna_db_protocol}{optuna_db_path}",
        load_if_exists=True,
        direction="minimize" if metric.lower_is_better else "maximize",
    )
    # Some predictors like GaussianNB do not take any hyperparameters, others are easily explored with
    # some categorical parameters. These therefore require 1 run or very few to explore the entire search
    # space. We therefore see if max_optuna_trials for the predictor has been reached in this study before
    # if so, then there is no reason to test those parameters again.
    if (
        predictor.max_optuna_trials is not None
        and len(optuna_study.trials) >= predictor.max_optuna_trials
    ):
        print(
            f"max_optuna_trials ({predictor.max_optuna_trials}) already reached for study, continuing."
        )
        return

    # It is a new study, set the user attributes for ease of analysis
    if len(optuna_study.trials) == 0:
        optuna_study.set_user_attr("phe_obj_name", phe_name)
        optuna_study.set_user_attr("predictor_name", predictor.name)
        optuna_study.set_user_attr("dataset", ":".join(dataset_combination))
        optuna_study.set_user_attr("prediction_type", prediction_type.name)
        if prediction_type == PredictionType.view and target_dataset_name is not None:
            optuna_study.set_user_attr("predicting_view", target_dataset_name)
        optuna_study.set_user_attr("metric_name", metric.name)
        optuna_study.set_user_attr("lower_is_better", metric.lower_is_better)
        optuna_study.set_user_attr("merge_folds", True)
    start_time = default_timer()
    optuna_study.optimize(
        lambda trial: _optuna_augmented_objective_merge_folds(
            trial,
            optuna_study,
            X,
            y,
            predictor,
            metric,
            n_splits,
            random_state=random_state,
        ),
        n_trials=n_optuna_trials_for_predictor,
    )
    optuna_study.set_user_attr("time_taken", default_timer() - start_time)
    test_score = metric(y_test, predictor._best_predictor_instance.predict(X_test))
    optuna_study.set_user_attr("test_score", test_score)


def _optuna_augmented_objective_merge_folds(
    trial: optuna.Trial,
    optuna_study: optuna.Study,
    X: Union[list[np.ndarray], np.ndarray],
    y: np.ndarray,
    predictor: PhenonautPredictor,
    metric: PhenonautPredictionMetric,
    n_splits: int = 5,
    random_state=None,
):
    if isinstance(random_state, np.random.Generator):
        random_state = random_state.integers(int(1e9))
    scores = []
    optuna_dict = {}
    if predictor.optuna is None:
        optuna_dict = {}
    else:
        optuna_dict = {
            hyperparameter.name: getattr(trial, hyperparameter.optuna_func)(
                hyperparameter.name, *hyperparameter.parameters
            )
            for hyperparameter in predictor.optuna
        }

    if predictor.conditional_hyperparameter_generator is not None:
        optuna_dict.update(
            predictor.conditional_hyperparameter_generator(
                predictor.conditional_hyperparameter_generator_constructor_keyword,
                trial,
                optuna_dict,
                X.shape[1],
                y.shape[1] if y.ndim == 2 else 1,
            )
        )

    if isinstance(predictor.predictor, tuple):
        predictor_instance = predictor.predictor[0](
            predictor.predictor[1](**optuna_dict)
        )
    else:
        predictor_instance = predictor.predictor(**optuna_dict)

    # multi view
    if isinstance(X, list):
        for train_indexes, validation_indexes in KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        ).split(X[0]):
            mv_X_train = [X[i][train_indexes] for i in range(len(X))]
            mv_X_test = [X[i][validation_indexes] for i in range(len(X))]
            predictor_instance.fit(mv_X_train, y[train_indexes])
            y_pred = predictor_instance.predict(mv_X_test)
            scores.append(metric(y[validation_indexes], y_pred))
    else:
        for train_indexes, validation_indexes in KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        ).split(X):
            predictor_instance.fit(X[train_indexes], y[train_indexes])
            y_pred = predictor_instance.predict(X[validation_indexes])
            scores.append(metric(y[validation_indexes], y_pred))

    # If not yet set (because its a new study with 0 trials and was initialised to None, then set it to the current score)
    if "best_trained_model_score" not in optuna_study.user_attrs:
        optuna_study.set_user_attr("best_trained_model_score", np.mean(scores))
    if not hasattr(predictor, "_best_predictor_instance"):
        predictor._best_predictor_instance = predictor_instance
    cur_best_score = optuna_study.user_attrs["best_trained_model_score"]

    if (metric.lower_is_better and np.mean(scores) <= cur_best_score) or (
        not metric.lower_is_better and np.mean(scores) >= cur_best_score
    ):
        optuna_study.set_user_attr("best_trained_model_score", np.mean(scores))
        predictor._best_predictor_instance = predictor_instance
    trial.set_user_attr("KFoldScores", scores)
    trial.set_user_attr("Mean_KFold", np.mean(scores))
    trial.set_user_attr("Std_KFold", np.std(scores))

    return np.mean(scores)
