# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from mvlearn.semi_supervised import CTClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from phenonaut.predict.predictor_dataclasses import (
    HyperparameterCategorical,
    HyperparameterFloat,
    HyperparameterInt,
    HyperparameterLog,
    PhenonautPredictor,
)

default_classifiers = [
    PhenonautPredictor("Naive Bayes", GaussianNB),
    PhenonautPredictor(
        "Random Forest",
        RandomForestClassifier,
        [
            HyperparameterCategorical("max_depth", (5, 10, 20, None)),
            HyperparameterInt("n_estimators", 50, 200),
            HyperparameterCategorical("max_features", (None, "sqrt", "log2")),
        ],
        constructor_kwargs={"n_jobs": -1},
    ),
    PhenonautPredictor(
        "KNN",
        KNeighborsClassifier,
        HyperparameterInt("n_neighbors", 2, 10),
        max_optuna_trials=9,
        constructor_kwargs={"n_jobs": -1},
    ),
    PhenonautPredictor(
        "Linear SVM",
        SVC,
        [
            HyperparameterCategorical("kernel", ("linear",)),
            HyperparameterLog("C", 0.001, 10),
            HyperparameterCategorical("gamma", ("scale", "auto")),
        ],
        dataset_size_cutoff=1000,
    ),
    PhenonautPredictor(
        "RBF SVM",
        SVC,
        [
            HyperparameterCategorical("kernel", ("rbf",)),
            HyperparameterLog("C", 0.001, 10),
            HyperparameterCategorical("gamma", ("scale", "auto")),
        ],
        dataset_size_cutoff=1000,
    ),
    PhenonautPredictor(
        "Gaussian Process",
        GaussianProcessClassifier,
        constructor_kwargs={"copy_X_train": False, "n_jobs": -1},
        embed_in_results=False,
    ),
    PhenonautPredictor(
        "Decision Tree",
        DecisionTreeClassifier,
        [
            HyperparameterCategorical("criterion", ("gini", "entropy")),
            HyperparameterCategorical("max_depth", (None, 5, 10, 15, 20)),
            HyperparameterCategorical("max_features", (None, "sqrt", "log2")),
        ],
    ),
    PhenonautPredictor(
        "SciKit NeuralNet",
        MLPClassifier,
        [
            HyperparameterLog("alpha", 0.00001, 0.001),
            HyperparameterCategorical("solver", ["adam", "sgd"]),
            HyperparameterCategorical("max_iter", [1000]),
            HyperparameterCategorical("learning_rate", ["constant", "adaptive"]),
        ],
    ),
    PhenonautPredictor("AdaBoost", AdaBoostClassifier),
    PhenonautPredictor("QDA", QuadraticDiscriminantAnalysis),
    PhenonautPredictor(
        "mvlearn CTClassifier (GNB,GNB)", CTClassifier, num_views=2, max_classes=2
    ),
]
