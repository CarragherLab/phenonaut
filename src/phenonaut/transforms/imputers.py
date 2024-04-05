from typing import Optional, Union

from sklearn.ensemble import RandomForestRegressor as _SKLearnRandomForestRegressor
from sklearn.experimental import (  # although it looks unused, it must be present to import iterative_imputer into sklearn.impute
    enable_iterative_imputer,
)
from sklearn.impute import IterativeImputer as _SKLearnIterativeImputer
from sklearn.impute import KNNImputer as _SKLearnKNNImputer
from sklearn.impute import SimpleImputer as _SKLearnSimpleImputer

from .transformer import Transformer


class SimpleImputer(Transformer):
    r"""SciKitLearn SimpleImputer

    SciKit's SimpleImputer wrapped in a Phenonaut Transformer. Allows passing a
    strategy argument containing 'mean', 'median', 'most_frequent', or 'constant'.
    If constant is passed, then must supply a fill_value argument.

    Can be used as follows:

    .. code-block:: python

        imputer=SimpleImputer()
        imputer(dataset)

    Parameters
    ----------
    strategy : str, optional
        The imputation strategy to use, can be 'mean', 'median', 'most_frequent',
        or 'constant', by default "median".
    fill_value : Optional[Union[str, float]], optional
        If constant is passed as the strategy, then this argument should contain
        the constant value to fill with. By default None.
    new_feature_names : Union[list[str], str], optional
        Name of new features. If ending in _, then this is prepended to existing
        features. By default 'Imputed\_'.
    """

    def __init__(
        self,
        strategy: str = "median",
        fill_value: Optional[Union[str, float]] = None,
        new_feature_names: Union[list[str], str] = "Imputed_",
    ):
        super().__init__(
            _SKLearnSimpleImputer,
            new_feature_names=new_feature_names,
            transformer_name="SimpleImputer",
            constructor_kwargs={"strategy": strategy, "fill_value": fill_value},
        )


class KNNImputer(Transformer):
    r"""SciKitLearn KNNImputer

    SciKit's KNNImputer wrapped in a Phenonaut Transformer. Allows passing a
    numer of neighbors to use for imputation.

    Can be used as follows:

    .. code-block:: python

        imputer=KNNImputer()
        imputer(dataset)

    Parameters
    ----------
    n_neighbors : int
        Use this many neighboring samples for imputation.
    weights : str
        Weight function for neighbor points. Can be one of 'uniform' (All points
        in neighborhood weighted equally), 'distance' (contribution of neighbor
        points by inverse of their distance), or a callable accepting distance
        matrix and returning an array of the same shape containing  weights, see
        https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
        for further info.
    new_feature_names : Union[list[str], str], optional
        Name of new features. If ending in _, then this is prepended to existing
        features. By default 'KNNImputed\_'.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        new_feature_names: Union[list[str], str] = "KNNImputed_",
    ):
        super().__init__(
            _SKLearnKNNImputer,
            new_feature_names=new_feature_names,
            transformer_name="KNNImputer",
            constructor_kwargs={"n_neighbors": n_neighbors, "weights": weights},
        )


class RFImputer(Transformer):
    r"""
    RandomForestImputer

    RandomForestImputer inspired by SKLearn documentation here:
    https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html#sphx-glr-auto-examples-impute-plot-iterative-imputer-variants-comparison-py

    Can be very computationally expensive, so careful setting of max_iter is advised.

    Can be used as follows:

    .. code-block:: python

        imputer=RFImputer()
        imputer(dataset)

    Parameters
    ----------
    rf_kwargs : dict
        Dictionary to use in constructing the SciKitLearn RandomForestRegressor,
        by default {'n_jobs':-1} (implying that all available processors should be used to
        fit the random forest regressor).  May take any values shown in sklearn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    max_iter : int
        The maximum number of times to impute the missing values and check convergence
        as dictated by tol.
    tol : float
        The tolerance at which convergence is accepted. Once changes between iterations
        are smaller than this value, then iteration stops, unless max_iter has been reached
        in which case, iteration stops before tol is reached.  By default 1e-3.
    new_feature_names : Union[list[str], str], optional
        Name of new features. If ending in _, then this is prepended to existing
        features. By default 'RFImputed\_'.
    """

    def __init__(
        self,
        rf_kwargs={"n_jobs": -1},
        max_iter: int = 25,
        tol: float = 1e-3,
        new_feature_names: Union[list[str], str] = "RFImputed_",
    ):
        super().__init__(
            _SKLearnIterativeImputer,
            new_feature_names=new_feature_names,
            transformer_name="RFImputer",
            constructor_kwargs={
                "random_state": 42,
                "estimator": _SKLearnRandomForestRegressor(**rf_kwargs),
                "max_iter": max_iter,
                "tol": tol,
            },
        )
