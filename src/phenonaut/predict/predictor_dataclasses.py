# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Optional, Type, Union

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_absolute_error
from dataclasses import field


@dataclass
class OptunaHyperparameter:
    """OptunaHyperparameter base dataclass which is inherited from"""

    name: str


@dataclass
class OptunaHyperparameterNumber(OptunaHyperparameter):
    """Optuna hyperparameter dataclass for numbers, inherited for int and float


    Raises
    ------
    ValueError
        Lower bound must be lower than upper bound.
    """

    lower_bound: int
    upper_bound: int
    needed: bool = True

    def _check_bounds(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"lower_bound must be lower than upper_bound "
                f"({self.lower_bound}!<{self.upper_bound}), consider a"
                "categorical hyperparameter with one option if you need"
                " a single choice optuna hyperparameter"
            )


@dataclass
class HyperparameterInt(OptunaHyperparameterNumber):
    """Optuna hyperparameter dataclass for ints"""

    def __post_init__(self):
        self._check_bounds()
        self.optuna_func = "suggest_int"
        self.parameters = (self.lower_bound, self.upper_bound)
        self.kwargs = {}


@dataclass
class HyperparameterFloat(OptunaHyperparameterNumber):
    """Optuna hyperparameter dataclass for floats"""

    def __post_init__(self):
        self._check_bounds()
        self.optuna_func = "suggest_float"
        self.parameters = (self.lower_bound, self.upper_bound)
        self.kwargs = {}


@dataclass
class HyperparameterLog(OptunaHyperparameterNumber):
    """Optuna hyperparameter dataclass for loguniform distributions of floats"""

    def __post_init__(self):
        self._check_bounds()
        self.optuna_func = "suggest_float"
        self.parameters = (self.lower_bound, self.upper_bound)
        self.kwargs = {'log': True}


@dataclass
class HyperparameterCategorical(OptunaHyperparameter):
    """Optuna hyperparameter dataclass for categorical lists"""

    choices: Union[list, tuple]
    needed: bool = True

    def __post_init__(self):
        if len(self.choices) == 0:
            raise ValueError(
                "Choices must be an iterable (usualy list or tuple) with length at least one"
            )
        self.optuna_func = "suggest_categorical"
        self.parameters = (self.choices,)
        self.kwargs = {}


@dataclass
class PhenonautPredictionMetric:
    """PhenonautPredictionMetric dataclass to hold metric, name and direction."""

    func: Callable
    name: str
    lower_is_better: bool

    def __call__(self, *args: Any, **kwds: Any) -> float:
        return self.func(*args, **kwds)


@dataclass(unsafe_hash=True)
class PhenonautPredictor:
    """PhenonautPredictor dataclass

    The PhenonautPredictor wraps classes with fit and predict methods,
    augmenting them with additional information like name, the number of views
    it may operate on at once, and hyperparameter lists which may be optimised
    using Optuna.
    """

    name: str
    predictor: Union[BaseEstimator, Callable]
    optuna: Optional[Union[Iterable[OptunaHyperparameter], OptunaHyperparameter]] = None
    num_views: int = 1
    max_optuna_trials: Optional[int] = None
    dataset_size_cutoff: Optional[int] = None
    constructor_kwargs: dict = field(default_factory=dict)
    max_classes: Optional[int] = None
    conditional_hyperparameter_generator_constructor_keyword: Optional[str] = (None,)
    conditional_hyperparameter_generator: Optional[Callable] = None
    embed_in_results: bool = True

    # Standardise self.optuna to an iterable if required.
    def __post_init__(self):
        if isinstance(self.optuna, OptunaHyperparameter):
            self.optuna = (self.optuna,)
        if self.optuna is None and self.max_optuna_trials is None:
            self.max_optuna_trials = 1
        if isinstance(self.optuna, Iterable):
            if all(
                [
                    isinstance(opt_option, HyperparameterCategorical)
                    for opt_option in self.optuna
                ]
            ):
                self.max_optuna_trials = reduce(
                    lambda x, y: x * y,
                    [len(opt_options.parameters) for opt_options in self.optuna],
                )
