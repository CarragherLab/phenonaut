# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from inspect import isclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from phenonaut.data.dataset import Dataset


class SupervisedTransformer:
    method = None
    method_kwargs = {}
    has_fit = False
    has_transform = False
    has_fit_transform = False
    new_feature_names = None
    is_callable = False
    callable_args = {}

    def __init__(
        self, method, new_feature_names=None, callable_args: dict = {}, **kwargs
    ):
        # Here we need to handle a method being put in, a class, and a class instance.

        if isclass(method):
            method = method(**kwargs)
        if hasattr(method, "fit"):
            self.has_fit = True
        if hasattr(method, "transform"):
            self.has_transform = True
        if hasattr(method, "fit_transform"):
            self.has_fit_transform = True
        if callable(method):
            self.is_callable = True
            self.method_kwargs = kwargs
            self.callable_args = callable_args
        self.method = method
        self.new_feature_names = new_feature_names

    def _fit_df(
        self, X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array]
    ):
        if not self.has_fit:
            raise TypeError(f"Supplied method {self.method} has no fit function")
        self.method.fit(X, y)

    def _transform_df(self, df: pd.DataFrame) -> np.ndarray:
        if self.is_callable:
            transformed_data = self.method(df, **self.callable_args)
            if isinstance(transformed_data, pd.DataFrame):
                return transformed_data.values
            return transformed_data

        if not self.has_transform:
            raise TypeError(f"Supplied method {self.method} has no transform function")
        transformed_data = self.method.transform(df)
        if isinstance(transformed_data, pd.DataFrame):
            return transformed_data.values
        return transformed_data

    def fit(
        self,
        data: Union[Dataset, pd.DataFrame],
        y_or_ycolumnlabel: Optional[Union[str, pd.Series, np.array]] = None,
        fit_perturbation_ids: Optional[Union[str, list]] = None,
    ):
        # Dataframe
        if isinstance(data, pd.DataFrame):
            if isinstance(y_or_ycolumnlabel, str):
                if y_or_ycolumnlabel in data.columns:
                    X = data[list(data.columns).remove(y_or_ycolumnlabel)]
                    y = data[y_or_ycolumnlabel]
                    self._fit_df(X, y)
            else:
                self._fit_df(data, y_or_ycolumnlabel)
        else:
            if not isinstance(data, Dataset):
                raise TypeError(
                    f"fit requires a pd.DataFrame, or a Dataset class, received {type(data)}"
                )
            if fit_perturbation_ids is None:
                fit_perturbation_ids = list(data.get_unique_perturbations())
            if isinstance(y_or_ycolumnlabel, str):
                self._fit_df(
                    data.df.loc[
                        data.df.query(
                            f"{data.perturbation_column} == @fit_perturbation_ids"
                        ).index,
                        data.features,
                    ],
                    data.df.loc[y_or_ycolumnlabel],
                )
            else:
                self._fit_df(
                    data.df.loc[
                        data.df.query(
                            f"{data.perturbation_column} == @fit_perturbation_ids"
                        ).index,
                        data.features,
                    ],
                    y_or_ycolumnlabel,
                )

    def transform(
        self,
        data: Union[Dataset, pd.DataFrame],
        new_feature_names: Optional[List[str]] = None,
    ):
        # Do appropriate calls for DataFrame or Dataset to get transformed_data
        # which will be a np.array
        transformed_data = None
        if isinstance(data, pd.DataFrame):
            transformed_data = self.transform_df(data.df)
        else:
            if not isinstance(data, Dataset):
                raise TypeError(
                    f"transform requires a pd.DataFrame, or a Dataset class, received {type(data)}"
                )
        transformed_data = self._transform_df(data.df.loc[:, data.features])

        # Now we assign features/column names with the following descending priority:
        # new_feature_names (argument to this function)
        # self.new_feature_names
        # generated from self.method.__name__ with -n after, where x is the column number/feature count.
        if new_feature_names is None:
            new_feature_names = self.new_feature_names
        if new_feature_names is None:
            new_feature_names = [
                f"{self.method.__name__}-{n+1}"
                for n in range(transformed_data.shape[1])
            ]
        data.df[new_feature_names] = transformed_data
        data.features = (new_feature_names, f"{self.method}")

    def _fit_transform_df(self, df: pd.DataFrame) -> np.ndarray:
        if not self.has_fit_transform:
            raise TypeError(
                f"Supplied method {self.method} has no fit_transform function"
            )

        transformed_data = self.method.fit_transform(df)
        if isinstance(transformed_data, pd.DataFrame):
            return transformed_data.values
        return transformed_data

    def fit_transform(
        self, data: Union[Dataset, pd.DataFrame], new_feature_names: list = None
    ):
        # Do appropriate calls for DataFrame or Dataset to get transformed_data
        # which will be a np.array
        transformed_data = None
        if isinstance(data, pd.DataFrame):
            transformed_data = self._fit_transform_df(data.df)
        else:
            if not isinstance(data, Dataset):
                raise TypeError(
                    f"fit_transform requires a pd.DataFrame, or a Dataset class, received {type(data)}"
                )
        transformed_data = self._fit_transform_df(data.df.loc[:, data.features])

        # Now we assign features/column names with the following descending priority:
        # new_feature_names (argument to this function)
        # self.new_feature_names
        # generated from self.method.__name__ with -n after, where x is the column number/feature count.
        if new_feature_names is None:
            new_feature_names = self.new_feature_names
        if new_feature_names is None:
            new_feature_names = [
                f"{self.method.__name__}-{n+1}"
                for n in range(transformed_data.shape[1])
            ]
        data.df[new_feature_names] = transformed_data
        data.features = (new_feature_names, f"{self.method}")
