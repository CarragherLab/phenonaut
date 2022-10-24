# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

"""Run example predict.profile on IRIS dataset

Uses fire to run classification, regression or view prediction (multiregression)
on the sklearn iris dataset.

Usage:
example0_iris.py COMMAND OUTPUT_DIR

COMMAND is one of the following:
    c (shortcut for classification)
    Run sklearn Iris example, predicting iris class from 1 and 2 views.

    classification
    Run sklearn Iris example, predicting iris class from 1 and 2 views.

    r (shortcut for regression)
    Run sklearn Iris example, predicting sepal length from 1 and 2 views.

    regression
    Run sklearn Iris example, predicting sepal length from 1 and 2 views.

    mr (shortcut for multiregression)
    Run sklearn Iris example, predicting sepal length from 1 and 2 views.

    multiregression
    Run sklearn Iris example, predicting sepal length from 1 and 2 views.

OUTPUT_DIR is the destination output directory
"""


import fire
import phenonaut
import phenonaut.predict
from sklearn.datasets import load_iris
from phenonaut.data import Dataset


def phenonaut_object_iris_2_views(iris_df, keep_only_class_1_and_2=False):
    if keep_only_class_1_and_2:
        iris_df = iris_df.query("target==0 or target==1")
    column_names = iris_df.columns.to_list()
    df1 = iris_df.iloc[:, [0, 1, 4]].copy()
    df2 = iris_df.iloc[:, [2, 3, 4]].copy()
    ds1 = Dataset("Iris_view1", df1, {"features": column_names[0:2]})
    ds2 = Dataset("Iris_view2", df2, {"features": column_names[2:4]})
    return phenonaut.Phenonaut([ds1, ds2], name="Iris dataset from scikit-learn")


def run_phenonaut_iris_classification(
    output_path: str,
    n_optuna_trials: int = 20,
    run_predictors: bool = True,
    optuna_merge_folds: bool = False,
):
    """Run sklearn Iris example, predicting iris class from 1 and 2 views.

    Parameters
    ----------
    output_path : Union[str, Path]
        Output directory for the results of phenonaut.predict.profile
    n_optuna_trials : int
        Number of optuna_trials to run for each predictor and each view, by default 20
    run_predictors : bool, optional
        If True, then run all predictors, by default True
    optuna_merge_folds : bool, optional
        By default, every fold of the train-validation split is optimised by optuna for a predictor&view pair.
        If optuna_merge_folds is true, then the average validation score of each fold is passed as the result
        to optuna, optimising models across all folds. , by default False.
    """
    phe = phenonaut_object_iris_2_views(load_iris(as_frame=True).frame, keep_only_class_1_and_2=True)
    phenonaut.predict.profile(
        phe,
        output_path,
        n_optuna_trials=n_optuna_trials,
        optuna_merge_folds=optuna_merge_folds,
        predictors=None if run_predictors else [],
    )


def run_phenonaut_iris_regression(
    output_path: str,
    n_optuna_trials: int = 20,
    run_predictors: bool = True,
    optuna_merge_folds: bool = False,
):
    """Run sklearn Iris example, predicting sepal length from 1 and 2 views.

    Parameters
    ----------
    output_path : Union[str, Path]
        Output directory for the results of phenonaut.predict.profile
    n_optuna_trials : int
        Number of optuna_trials to run for each predictor and each view, by default 20
    run_predictors : bool, optional
        If True, then run all predictors, by default True
    optuna_merge_folds : bool, optional
        By default, every fold of the train-validation split is optimised by optuna for a predictor&view pair.
        If optuna_merge_folds is true, then the average validation score of each fold is passed as the result
        to optuna, optimising models across all folds. , by default False.
    """
    phe = phenonaut.Phenonaut(load_iris(), dataframe_name="Iris, predict sepal length")
    target_column_name = "sepal length (cm)"
    y = phe[0].df[target_column_name]
    phe[0].drop_columns([target_column_name, "target"])
    phenonaut.predict.profile(
        phe,
        output_path,
        n_optuna_trials=n_optuna_trials,
        optuna_merge_folds=optuna_merge_folds,
        predictors=None if run_predictors else [],
        target=y,
    )



def run_phenonaut_iris_multiregression(
    output_path: str,
    n_optuna_trials: int = 20,
    run_predictors: bool = True,
    optuna_merge_folds: bool = False,
):
    """Run sklearn Iris example, predicting sepal length from 1 and 2 views.

    Parameters
    ----------
    output_path : Union[str, Path]
        Output directory for the results of phenonaut.predict.profile
    n_optuna_trials : int
        Number of optuna_trials to run for each predictor and each view, by default 20
    run_predictors : bool, optional
        If True, then run all predictors, by default True
    optuna_merge_folds : bool, optional
        By default, every fold of the train-validation split is optimised by optuna for a predictor&view pair.
        If optuna_merge_folds is true, then the average validation score of each fold is passed as the result
        to optuna, optimising models across all folds. , by default False.
    """
    phe=phenonaut_object_iris_2_views(load_iris(as_frame=True).frame, keep_only_class_1_and_2=True)
    phe[0].drop_columns("target")
    phe[1].drop_columns("target")
    
    phenonaut.predict.profile(
        phe,
        output_path,
        n_optuna_trials=n_optuna_trials,
        optuna_merge_folds=optuna_merge_folds,
        predictors=None if run_predictors else [],
        dataset_combinations=((0,),),
        target=phe[1]
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "c": run_phenonaut_iris_classification,
            "classification": run_phenonaut_iris_classification,
            'r': run_phenonaut_iris_regression,
            'regression': run_phenonaut_iris_regression,
            'mr': run_phenonaut_iris_multiregression,
            'multiregression': run_phenonaut_iris_multiregression,
        }
    )
