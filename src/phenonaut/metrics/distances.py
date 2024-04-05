# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.errors import DataError
from scipy.linalg import inv
from scipy.spatial import distance
from scipy.spatial.distance import cdist, cityblock, pdist
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from phenonaut.data import Dataset, dataset


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
        raise DataError(
            "Dataset does not have a perturbation column set and none was supplied to treatment_spread_euclidean"
        )
    if perturbation_column is None:
        perturbation_column = data.perturbation_column
    perturbations = (
        data.get_unique_perturbations() if perturbations is None else perturbations
    )

    return {
        t: np.sum(
            pdist(
                dataset.df.query(f"{data.perturbation_column} == '{t}'")[data.features]
            ),
            axis=None,
        )
        for t in perturbations
    }


def mahalanobis(
    point: Union[List[float], np.ndarray, pd.DataFrame],
    cloud: Union[List[List[float]], np.ndarray, pd.DataFrame],
    pvals: bool = False,
    covariance: Union[
        np.ndarray, EmpiricalCovariance, MinCovDet, None
    ] = EmpiricalCovariance(),
):
    """Measure the Mahalanobis distance between a point and a cloud

    The Mahalanobis distance calculation is particularly sensitive to outliers,
    which results in large changes in covariance matrix calculation. For this
    reason, robust covarience estimators may be suppplied to the method. Whilst
    a common recomendation is to calculate the sqrt of the Mahalanobis distance,
    this is an approximation to euclidean space, only correct when the covariance
    matrix is an identity matrix. As Phenonaut is concerned on operating on high
    dimensional space which very likely has covariences present, the returned
    distance is not square rooted, returning D2 as noted.

    https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/euclid

    Optionally, the p-value for the point or points belonging to the cloud can
    be returned.

    Parameters
    ----------
    point : Union[List[float],np.ndarray, pd.DataFrame]
        Multidimensional point, can be a simple list of features, or a 2d M*N
        list where M are multiple points to be individually measured of N
        features and returning an array of measurmenents for each point.
    cloud : Union[List[List[float]], np.ndarray, pd.DataFrame]
        2D M*N array-like set of M points, with N features from which the
        underlying target distribution will be measured.
    pvals : bool
        If True, then p-value for the point (or points) belonging to cloud is returned.
        This is calculated using Chi2 and degrees of freedom calculated as
        N-1, where N is the number of features/dimensions. If point was one
        dimensional, then a single floating point value is returned. If it is 2D
        (MxN matrix), then an array of length M is returned. By default, False.
    covariance : Optional[Union[np.ndarray, EmpiricalCovariance]], optional
        If none, then the covariance matrix is calculated using scikit's
        EmpiricalCovariance. This is fairly robust to outliers, and much more robust
        than the standard approach to calculating a covariance matrix (using numpy's
        np.cov method. Robust estimators may be used by passing in an instantiated
        object which a subclass of EmpiracalCovariance (like
        sklearn.covariance.MinCovDet or EmpiricalCovariance itself). If None, then
        numpy's cov method is used (sensitive to outliers). By default EmpiracalCovariance().

    Returns
    -------
    [np.ndarray, float]
        Array (if point is 2D) of distances. If 1D, then single float value is
        returned indicating the Mahalanobis distance of the point to the
        cloud.

    """
    point = np.array(point)
    cloud = np.array(cloud)
    if point.ndim == 1:
        point = point.reshape(1, -1)

    if isinstance(covariance, EmpiricalCovariance):
        covariance.fit(cloud)
        mahl_d = covariance.mahalanobis(point)
    else:
        if covariance is None:
            cov_mat = np.cov(cloud.T)
        elif isinstance(covariance, np.ndarray):
            cov_mat = covariance
        else:
            raise ValueError(
                f"cov argument should be one of np.ndarray, None, or a subclass of sklearn.covariance.EmpiricalCovariance, it was {type(covariance)}"
            )
        difference = point - np.mean(cloud, axis=0)
        mahl_d = np.diag(
            np.dot(np.dot(difference, np.linalg.inv(cov_mat)), difference.T)
        )
    if len(mahl_d) == 1:
        mahl_d = mahl_d[0]
    if not pvals:
        return mahl_d
    else:
        from scipy.stats import chi2

        return 1 - chi2.cdf(mahl_d, cloud.shape[1] - 1)


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
