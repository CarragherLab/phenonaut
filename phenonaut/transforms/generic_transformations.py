# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from abc import ABC
from typing import Optional, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from phenonaut.data import Dataset
import numpy as np


class GenericTransform(ABC):
    def __init__(self):
        pass

    def __call__(
        self,
    ):
        pass


class StandardScaler(GenericTransform):
    def __init__(self):
        """StandardScaler
        
        Perform StandardScaler on a dataset
        
        May be instantiated and then called, or fit and then transform performed.
        Optional query argument carries out a pandas query to select relevant rows
        from the dataframe to perform the fit.

        """
        self._sc=sklearn_StandardScaler()
    def fit(self, dataset:Dataset, query:Optional[str]=None):
        """Fit the standard scaler to a dataset

        Parameters
        ----------
        dataset : Dataset
            Dataset upon which the fit is to be performed
        query : Optional[str], optional
            Pandas query which may be used to select rows, by default None

        Returns
        -------
        StandardScaler
            Returns a reference to itself as an object.
        """        
        if query is None:
            self._sc.fit(dataset.data)
        else:
            self._sc.fit(dataset.df.query(query)[dataset.features])
        return self
    def transform(self, dataset:Dataset):
        """Perform transform

        Parameters
        ----------
        dataset : Dataset
            The dataset upon which the transformation should be applied.
        """        
        dataset.df[dataset.features] = self._sc.fit_transform(
            dataset.data
        )
        dataset.features=dataset.features, "Applied standard scalar"
    def __call__(self, dataset: Dataset, query:Optional[str]=None):
        """Perform StandardScaler

        Object may be called directly to perform fit and then transform.
        Parameters
        ----------
        dataset : Dataset
            The dataset upon which the transformation should be applied.
        query : Optional[str], optional
            Pandas query which may be used to select rows, by default None
        """        
        self.fit(dataset, query)
        self.transform(dataset)


class Log2(GenericTransform):
    def __call__(self, dataset: Dataset):
        super().__init__()
        dataset.df[dataset.features] = np.log2(dataset.data)

class RobustMAD(GenericTransform):
    def __init__(self, mad_scale:Union[str, float]='normal', epsilon:Optional[float]=None):
        """RobustMAD
        
        Robust scaling, as used popularly in pycytominer

        https://github.com/cytomining/pycytominer/blob/master/pycytominer/operations/transform.py

        Perform robust normalization using median and mad
        (median absolute deviation), like:
        
        X=X-median/(mad+epsilon)
        
        May be instantiated and then called, or fit and then transform performed.
        Optional query argument carries out a pandas query to select relevant rows
        from the dataframe to perform the fit.

        Parameters
        ----------
        mad_scale : Union[str,float], optional
            Scaling argument passed to the median_abs_deviation 
            scipy function. The value 'normal' implies the data
            is normally distributed, but this may be changed
            for other distribution types. See scipy documentation
            for the function for a more complete explanation.
            By default 'normal'.
        epsilon : float, optional
            Small fudge-factor to remove danger of median value.
            If None, then it is set by numpy eps value.
            By default None.
        """        
        self._epsilon=np.finfo(np.float64).eps if epsilon is None else epsilon
        self._mad_scale=mad_scale

    def fit(self, dataset:Dataset, query:Optional[str]=None):
        """Fit the RobustMAD scaler to a dataset

        Parameters
        ----------
        dataset : Dataset
            Dataset upon which the fit is to be performed
        query : Optional[str], optional
            Pandas query which may be used to select rows, by default None

        Returns
        -------
        RobustMAD
            Returns a reference to itself as an object.
        """        
        from scipy.stats import median_abs_deviation
        if query is None:
            self._median=dataset.data.median()
            self._mad=pd.Series(median_abs_deviation(dataset.data, nan_policy="omit", scale=self._mad_scale), index=self._median.index)
        else:
            query_df=dataset.df.query(query)[dataset.features]
            self._median=query_df.median()
            self._mad=pd.Series(median_abs_deviation(query_df, nan_policy="omit", scale=self._mad_scale), index=self._median.index)
        return self
    def transform(self, dataset:Dataset):
        """Perform transform

        Parameters
        ----------
        dataset : Dataset
            The dataset upon which the transformation should be applied.
        """        
        dataset.df[dataset.features]=(dataset.data-self._median)/(self._mad+self._epsilon)
    def __call__(self, dataset:Dataset, query:Optional[str]=None):
        """Perform RobustMAD

        Object may be called directly to perform fit and then transform.
        Parameters
        ----------
        dataset : Dataset
            The dataset upon which the transformation should be applied.
        query : Optional[str], optional
            Pandas query which may be used to select rows, by default None
        """        
        self.fit(dataset, query)
        self.transform(dataset)

