# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from typing import List, Tuple, Union, Optional

from pandas.errors import DataError

from phenonaut.data import Dataset, dataset
import pandas as pd

from scipy.spatial.distance import pdist, cdist, cityblock
from scipy.linalg import inv
import numpy as np
from scipy.spatial import distance
from sklearn.covariance import MinCovDet


def treatment_spread_euclidean(
    data: Dataset,
    perturbation_column: Optional[str] = None,
    perturbations: Optional[List[str]] = None,
) -> dict[str:float]:
    """Calculate the euclidean spreat of perturbation repeats

    Returns
    -------
    dict[str:float]
        Dictionary with perturbations as keys and euclidean distances as values.

    Raises
    ------
    DataError
        Perturbation column was not set for the given Dataset and was not
        supplied via the perturbation_column argument to
        treatment_spread_euclidean.
    """    
    if perturbation_column is None and data.perturbation_column is None:
        raise DataError("Dataset does not have a perturbation column set and none was supplied to treatment_spread_euclidean")
    if perturbation_column is None:
        perturbation_column=data.perturbation_column
    perturbations = data.get_unique_perturbations() if perturbations is None else perturbations
    
    return {
        t: np.sum(
            pdist(dataset.df.query(f"{data.perturbation_column} == '{t}'")[data.features]),
            axis=None,
        )
        for t in perturbations
    }


def mahalanobis(
    point: Union[List[float], np.ndarray, pd.DataFrame],
    cloud: Union[List[List[float]], np.ndarray, pd.DataFrame],
    covariance_estimator: Optional[object] = MinCovDet(),
):
    """Measure the Mahalanobis distance between a point and a cloud

    Parameters
    ----------
    point : Union[List[float],np.ndarray, pd.DataFrame]
        Multidimensional point, can be a simple list of [x,y,z], or a 2d M*N
        list where M are multiple points to be individually measured of N
        features and returning an array of measurmenents for each point.
    cloud : Union[List[List[float]], np.ndarray, pd.DataFrame]
        2D M*N array-like set of M points, with N features from which the
        underlying target distribution will be measured.
    covariance_estimator : Optional[object], optional
        Instantiated object with fit and mahalanobis methods used for covariance
        estimation. MinCovDet is used by default from sklearn.covariance.
        Another option is sklearn.covariance.EmpiricalCovariance().

    Returns
    -------
    [np.ndarray, float]
        Array (if point is 2D) of distances. If 1D, then single float value is
        returned indicating the Mahalanobis distance of the point to the
        cloud.

    Raises
    ------
    DataError
        Point can be 2D, but not 3D.
    """
    point = np.array(point)
    if point.ndim == 1:
        point = point.reshape(1, -1)

    if point.ndim > 2:
        raise DataError(
            f"point can be a 2D M*N array where M is unique points, and N are features. Point shape was {point.shape}."
        )
    covariance_estimator.fit(cloud.values)
    results = np.empty(point.shape[0])
    # for p,i in enumerate(point):
    results = covariance_estimator.mahalanobis(point)
    results = np.sqrt(results)
    if len(results) == 1:
        return np.sqrt(results[0])
    else:
        return results


def euclidean(
    point1: Union[List[float], np.ndarray, pd.DataFrame],
    point2: Union[List[List[float]], np.ndarray, pd.DataFrame],
):
    """Measure the euclidean distance between 2 points

    Parameters
    ----------
    point1 : Union[List[float],np.ndarray, pd.DataFrame]
        Multidimensional point, can be a simple list of [x,y,z], or a 2d M*N
        list where M are multiple points to be individually measured of N
        features and returning an array of measurmenents for each point.
    point2 : Union[List[List[float]], np.ndarray, pd.DataFrame]
        List of N features, must have 1 dimension.

    Returns
    -------
    [np.ndarray, float]
        Array (if point is 2D) of distances. If 1D, then single float value is
        returned indicating the euclidean distance between point1 and point2.

    Raises
    ------
    DataError
        Point1 can be 2D, but not 3D, and point2 must be 1D
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    if point1.ndim == 1:
        point1 = point1.reshape(1, -1)
    if point1.ndim > 2:
        raise DataError(
            f"point1 can be a 2D M*N array where M is unique points, and N are features. point1 shape was {point1.shape}."
        )
    if point2.ndim > 1:
        raise DataError(f"point2 must be 1D, it was {point2.ndim}.")
    results = np.empty(point1.shape[0])
    for i, p in enumerate(point1):
        results[i] = np.linalg.norm(point2 - p)
    if len(results) == 1:
        return np.sqrt(results[0])
    else:
        return results


def manhattan(
    point1: Union[List[float], np.ndarray, pd.DataFrame],
    point2: Union[List[List[float]], np.ndarray, pd.DataFrame],
):
    """Measure the Manhattan distance between 2 points

    Parameters
    ----------
    point1 : Union[List[float],np.ndarray, pd.DataFrame]
        Multidimensional point, can be a simple list of [x,y,z], or a 2d M*N
        list where M are multiple points to be individually measured of N
        features and returning an array of measurmenents for each point.
    point2 : Union[List[List[float]], np.ndarray, pd.DataFrame]
        List of N features, must have 1 dimension.

    Returns
    -------
    [np.ndarray, float]
        Array (if point is 2D) of distances. If 1D, then single float value is
        returned indicating the euclidean distance between point1 and point2.

    Raises
    ------
    DataError
        Point1 can be 2D, but not 3D, and point2 must be 1D
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    if point1.ndim == 1:
        point1 = point1.reshape(1, -1)
    if point1.ndim > 2:
        raise DataError(
            f"point1 can be a 2D M*N array where M is unique points, and N are features. point1 shape was {point1.shape}."
        )
    if point2.ndim > 1:
        raise DataError(f"point2 must be 1D, it was {point2.ndim}.")
    results = np.empty(point1.shape[0])
    for i, p in enumerate(point1):
        results[i] = cityblock(p, point2)
    if len(results) == 1:
        return np.sqrt(results[0])
    else:
        return results
