# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import fire

from phenonaut import Phenonaut, packaged_datasets, predict


def run_phenonaut_tcga(
    output_path: str,
    tcga_path: str,
    n_optuna_trials: int = 20,
    run_predictors: bool = True,
    optuna_merge_folds: bool = False,
):
    """Run TCGA example, predicting one year survival rate over a range of datasets and combinations

    Parameters
    ----------
    output_path : Union[str, Path]
        Output directory for the results of phenonaut.predict.profile.
    tcga_path : str
        The location of the TCGA dataset if already downloaded, otherwise, the destination location.
    n_optuna_trials : int
        Number of optuna_trials to run for each predictor and each view, by default 20.
    run_predictors : bool, optional
        If True, then run all predictors, by default True.
    optuna_merge_folds : bool, optional
        By default, every fold of the train-validation split is optimised by optuna for a predictor&view pair.
        If optuna_merge_folds is true, then the average validation score of each fold is passed as the result
        to optuna, optimising models across all folds. , by default False.
    """
    phe = Phenonaut(
        dataset=packaged_datasets.TCGA(
            root=tcga_path, prediction_target="survives_1_year", download=True
        )
    )
    predict.profile(
        phe,
        output_path,
        metric="AUROC",
        n_optuna_trials=n_optuna_trials,
        predictors=None if run_predictors else [],
        optuna_merge_folds=optuna_merge_folds,
    )


if __name__ == "__main__":
    fire.Fire(run_phenonaut_tcga)
