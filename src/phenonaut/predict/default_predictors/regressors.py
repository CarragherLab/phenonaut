# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from phenonaut.predict.predictor_dataclasses import (
    HyperparameterCategorical,
    HyperparameterFloat,
    HyperparameterInt,
    HyperparameterLog,
    PhenonautPredictor,
)

default_regressors = [
    PhenonautPredictor(
        "Random Forest",
        RandomForestRegressor,
        [
            HyperparameterCategorical("max_depth", [5, 10, 20, None]),
            HyperparameterInt("n_estimators", 50, 200),
            HyperparameterCategorical("max_features", (None, "sqrt", "log2")),
        ],
        constructor_kwargs={"n_jobs": -1},
    ),
    PhenonautPredictor(
        "Linear SVR",
        SVR,
        [
            HyperparameterCategorical("kernel", ["linear"]),
            HyperparameterLog("C", 0.001, 10),
            HyperparameterCategorical("gamma", ["scale", "auto"]),
        ],
        dataset_size_cutoff=1000,
    ),
    PhenonautPredictor(
        "RBF SVR",
        SVR,
        [
            HyperparameterCategorical("kernel", ["rbf"]),
            HyperparameterLog("C", 0.001, 10),
            HyperparameterCategorical("gamma", ["scale", "auto"]),
        ],
        dataset_size_cutoff=1000,
    ),
    PhenonautPredictor(
        "AdaBoost Regressor",
        AdaBoostRegressor,
        HyperparameterInt("n_estimators", 50, 200),
    ),
    PhenonautPredictor("Gradient Boosting Regressor", GradientBoostingRegressor),
    PhenonautPredictor("Decision Tree Regressor", DecisionTreeRegressor),
    PhenonautPredictor(
        "Gaussian Process Regressor",
        GaussianProcessRegressor,
        constructor_kwargs={"copy_X_train": False},
        embed_in_results=False,
    ),
    PhenonautPredictor(
        "KNeighbors Regressor",
        KNeighborsRegressor,
        HyperparameterInt("n_neighbors", 2, 10),
        constructor_kwargs={"n_jobs": -1},
    ),
    PhenonautPredictor(
        "SciKit NeuralNet Regressor",
        MLPRegressor,
        [
            HyperparameterLog("alpha", 0.00001, 0.01),
            HyperparameterCategorical("solver", ["adam", "sgd"]),
            HyperparameterCategorical("max_iter", [1000]),
            HyperparameterCategorical("learning_rate", ["constant", "adaptive"]),
        ],
    ),
]
