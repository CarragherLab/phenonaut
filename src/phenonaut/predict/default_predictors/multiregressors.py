# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from phenonaut.predict.default_predictors.pytorch_models import (
    DAVE,
    MultiRegressorNN,
)
from phenonaut.predict.predictor_dataclasses import (
    HyperparameterCategorical,
    HyperparameterFloat,
    HyperparameterInt,
    HyperparameterLog,
    PhenonautPredictor,
)

default_multiregressors = [
    PhenonautPredictor(
        "DummyRegressor-Average", (MultiOutputRegressor, DummyRegressor)
    ),
    PhenonautPredictor(
        "MultiOutputRegressor-LinearRegression",
        (MultiOutputRegressor, LinearRegression),
    ),
    PhenonautPredictor(
        "MultiOutputRegressor-GradientBoosting",
        (MultiOutputRegressor, GradientBoostingRegressor),
        [
            HyperparameterFloat("learning_rate", 0.05, 0.2),
            HyperparameterInt("n_estimators", 1, 200),
        ],
    ),
    PhenonautPredictor(
        "MultiRegressorNN",
        MultiRegressorNN,
        [
            HyperparameterLog("learning_rate", 1e-5, 1e-1),
            HyperparameterInt("epochs", 1, 50),
            HyperparameterInt("num_hidden_layers", 1, 10),
            HyperparameterCategorical("use_optimizer", ["ADAM", "SGD"]),
        ],
        # conditional_hyperparameter_generator_constructor_keyword="hidden_layer_sizes",
        # conditional_hyperparameter_generator=lambda trial, optuna_dict, num_features_in, num_features_out: [
        #     trial.suggest_int(
        #         f"nodes_in_hidden_{n+1}",
        #         min(num_features_in, num_features_out),
        #         max(num_features_in, num_features_out) * 2,
        #     )
        #     for n in range(optuna_dict["num_hidden_layers"])
        # ],
    ),
    PhenonautPredictor(
        "DAVE",
        DAVE,
        [
            HyperparameterInt("num_hidden_layers", 1, 10),
            HyperparameterLog("learning_rate", 1e-5, 1e-1),
            HyperparameterCategorical("batch_size", [64, 128, 256, 512, 1024]),
            HyperparameterInt("epochs", 1, 20),
        ],
    ),
]
