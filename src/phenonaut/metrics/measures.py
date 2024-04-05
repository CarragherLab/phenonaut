# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from typing import Optional, Union

import numpy as np
from pandas import DataFrame

from phenonaut.data import Dataset
from phenonaut.phenonaut import Phenonaut


def _scalar_projection_and_rejection(a: np.ndarray, b: np.ndarray):
    """Return two scalar projection and rejection lists of a(s) onto b.

    Scalar projection and rejection calculated from:
    https://en.wikipedia.org/w/index.php?title=Scalar_projection&oldid=1038765889

    Projection of a onto unit vector of b (scalar projection) gets the length of
    components of a which are along b. Scalar rejection is therefore a-b,
    capturing components of a which are not along unit vector b and returning
    the length.  b is typically the target phenotype to which on off- values for
    phenotypes in a are assigned.

    a can be MxN numpy list, whereby M samples with N features have their scalar
    projections and rejections calculated, and returned as two arrays.
    Alternatively, a can be a 1D array capturing one sigle row of phenotypic
    features.

    Parameters
    ----------
    a : np.ndarray
        Query vector
    b : np.ndarray
        Target vector

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Returns two numpy arrays, the first being a list of scalar projection
        values for M samples and the second being the scalar rejection values
        for the M samples.
    """
    # Check input is an array, as if a list, this can cause issues whereby [1,2]*2
    # produces a longer list, not multiplying elements, also we want to iterate
    # rows, so if 1d, change to 2D.
    input_shape = a.shape
    a = np.array(a)
    b = np.array(b)
    if len(input_shape) == 1:
        a = a.reshape(1, -1)
    scalar_projection_values = np.empty(a.shape[0])
    scalar_rejection_values = np.empty(a.shape[0])

    # We need a unit vector of B - strange that nympy does not have a single
    # function to do this.
    unit_vec = lambda x: x / np.linalg.norm(x)

    for i, row in enumerate(a):
        scalar_projection_values[i] = np.dot(row, unit_vec(b))

        # a1 (using the nomenclarute of the wikipedia page) ius components of A which
        # are not along unit vector b.
        a1 = np.dot(row, b) / (np.dot(b, b)) * b
        scalar_rejection_values[i] = np.linalg.norm(row - a1)

    if len(input_shape) == 1:
        return scalar_projection_values.item(0), scalar_rejection_values.item(0)
    return (scalar_projection_values, scalar_rejection_values)


def _scalar_projection(a: np.ndarray, b: np.ndarray):
    """Return scalar projection of a(s) onto b

    Calls _scalar_projection_and_rejection and takes the first returned tuple
    value.

    See _scalar_projection_and_rejection for further information

    Parameters
    ----------
    a : np.ndarray
        Query vector
    b : np.ndarray
        Target vector

    Returns
    -------
    Union[np.ndarray, float]
        Returns a numpy array or single float value giving the scalar
        projection
        values for M samples and the second being the scalar rejection values
        for the M samples.
    """
    return _scalar_projection_and_rejection(a, b)[0]


def _scalar_rejection(a: np.ndarray, b: np.ndarray):
    """Return scalar rejection of a(s) onto b.

    Calls _scalar_projection_and_rejection and takes the second returned tuple
    value.

    See _scalar_projection_and_rejection for further information

    Parameters
    ----------
    a : np.ndarray
        Query vector
    b : np.ndarray
        Target vector

    Returns
    -------
    Union[np.ndarray, float]
        Returns a numpy array or single float value giving the scalar
        projection
        values for M samples and the second being the scalar rejection values
        for the M samples.
    """
    return _scalar_projection_and_rejection(a, b)[1]


def _scalar_projection_scaled(a: np.ndarray, b: np.ndarray):
    """Return scalar projection of a(s) onto b, normalised to the length of b

    Calls _scalar_projection_and_rejection and takes the first returned tuple
    value.

    See _scalar_projection_and_rejection for further information

    Parameters
    ----------
    a : np.ndarray
        Query vector
    b : np.ndarray
        Target vector

    Returns
    -------
    Union[np.ndarray, float]
        Returns a numpy array or single float value giving the scalar
        projection
        values for M samples and the second being the scalar rejection values
        for the M samples.
    """
    return _scalar_projection_and_rejection(a, b)[0] / np.linalg.norm(b)


def _scalar_rejection_scaled(a: np.ndarray, b: np.ndarray):
    """Return scalar rejection of a(s) onto b, normalised to the length of b

    Calls _scalar_projection_and_rejection and takes the second returned tuple
    value.

    See _scalar_projection_and_rejection for further information

    Parameters
    ----------
    a : np.ndarray
        Query vector
    b : np.ndarray
        Target vector

    Returns
    -------
    Union[np.ndarray, float]
        Returns a numpy array or single float value giving the scalar
        projection
        values for M samples and the second being the scalar rejection values
        for the M samples.
    """
    return _scalar_projection_and_rejection(a, b)[1] / np.linalg.norm(b)


def scalar_projection(
    dataset: Dataset,
    target_perturbation_column_name="control",
    target_perturbation_column_value="pos",
    output_column_label="pos",
    norm=True,
):
    """Calculates the scalar projection and scalar rejection, quantifying on and
        off target phenotypes, as used in:
        Heiser, Katie, et al. "Identification of potential treatments for
        COVID-19 through artificial intelligence-enabled phenomic analysis of
        human cells infected with SARS-CoV-2." BioRxiv (2020).

    Parameters
    ----------
    dataset : Dataset
        Phenonaut dataset being queries
    target_TreatmentID : [type]
        TreatmentID of target phenotype. The median of all features across wells
        containing this phenotype is used.
    """

    on_phenotype_feature_label = f"on_target_{output_column_label}"
    off_phenotype_feature_label = f"off_target_{output_column_label}"

    # Target phenotype, against which all else is measured, calculated as the
    # median of all features from all wells carrying target phenotype perturbation.
    median_target_phenotype = dataset.df.loc[
        dataset.df.eval(
            f"{target_perturbation_column_name} == '{target_perturbation_column_value}'"
        ),
        dataset.features,
    ].median(axis=0)

    if median_target_phenotype.isnull().values.any():
        raise ValueError(
            f"Target phenotype contained null values: {median_target_phenotype}"
        )

    # a is a point in phenotypic space - actually the entire dataframe as all is
    # calculated in one call.
    a = dataset.data
    # get scalar projection and rejection values.
    (
        scalar_projection_values,
        scalar_rejection_values,
    ) = _scalar_projection_and_rejection(a, median_target_phenotype)
    dataset.df[on_phenotype_feature_label] = scalar_projection_values / np.linalg.norm(
        median_target_phenotype
    )
    dataset.df[off_phenotype_feature_label] = scalar_rejection_values / np.linalg.norm(
        median_target_phenotype
    )

    dataset.features = (
        [on_phenotype_feature_label, off_phenotype_feature_label],
        f"Applied ScalarProjection in comparison with median phentype of {output_column_label}",
    )


# corrcoef_features_to_target
def feature_correlation_to_target(
    dataset: Union[Dataset, Phenonaut, DataFrame],
    target_feature: str,
    features: Optional[list[str]] = None,
    method: str = "pearson",
    return_dataframe: bool = True,
):
    """
    Calculate correlation coefficients for features to a column

    Sometimes we may wish to identify highly correlated features with a given
    property.

    In this example, we use a subset of the Iris dataset:

    ================= ================ ================= ================ ======
    sepal length (cm) sepal width (cm) petal length (cm) petal width (cm) target
    ================= ================ ================= ================ ======
    5.4                3.4               1.7              0.2              0
    7.2                3.0               5.8               1.6             2
    6.4                2.8               5.6               2.1             2
    4.8                3.1               1.60               2              0
    5.6                2.5               3.9               1.1             1
    ================= ================ ================= ================ ======

    We may wish to determine which feature is correlated with the
    "petal length (cm)" which can be achieved through calling this
    feature_correlation_to_target function, alowing the return of a pd.DataFrame
    or simple dictionary containing features names as keys, and the coefficients
    as values.

    .. code-block:: python

        import tempfile
        from phenonaut import Phenonaut
        with tempfile.NamedTemporaryFile(mode = "w") as tmp:
            tmp.write("sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target\\n5.4,3.4,1.7,0.2,0\\n7.2,3.0,5.8,1.6,2\\n6.4,2.8,5.6,2.1,2\\n4.8,3.1,1.60,2,0\\n5.6,2.5,3.9,1.1,1\\n")
            tmp.flush()
            phe=Phenonaut()
            phe.load_dataset("Flowers",tmp.name, {'features_regex':".*(width|length).*"})

        from phenonaut.metrics import feature_correlation_to_target
        print(feature_correlation_to_target(phe, 'petal length (cm)'))

    Returns a pd.DataFrame containing correlation coefficients:

    ================= ================================
    index             correlation_to_petal length (cm)
    ================= ================================
    sepal length (cm) 0.914639
    petal width (cm)  0.448289
    sepal width (cm)  -0.544665
    ================= ================================

    The optional dictionary, returned by calling the function with the additional return_dataframe parameter set to False:

    .. code-block:: python

        print(feature_correlation_to_target(phe, 'petal length (cm)'), return_dataframe=False)

    has the form:

    .. code-block:: python

        {'petal width (cm)': 0.448289248746271, 'sepal length (cm)': 0.9146393603234955, 'sepal width (cm)': -0.5446646166252519}

    Parameters
    ----------
    dataset : Union[Dataset, Phenonaut, DataFrame]
        The Phenonaut Dataset, pd.DataFrame, or Phenonaut object (containing only
        one dataset) on which to perform correlation calculations.
    target_feature : str
        The feature, or metadata column that all correlations should be calculated against.
    features : Optional[list[str]], optional
        List of features to include in the correlation calculations. If None,
        and a Dataset is supplied then those features are used. In the case where
        a pd.DataFrame is supplied, then features must be supplied. By default None.
    method : str, optional
        Method used to calculate the correlation coefficient. Can be 'pearson',
        'kendall' for the Kendall Tau correlation coefficient, or 'spearman' for
        the Spearman rank correlation. By default "pearson".
    return_dataframe : bool, optional
        If True, then a pd.DataFrame containing correlations is returned. If False,
        then a dictionary is returned, containing features names as keys, and
        the coefficients as values. By default True.

    Returns
    -------
    Union[pd.Datframe, dict]
        Return a pd.DataFrame with calculated correlation coefficients, or
        alternatively, a dictionary containing features names as keys, and the
        coefficients as values.

    Raises
    ------
    ValueError
        Phenonaut objects must contain only one dataset if passed to this
        function.
    ValueError
        Target feature not found in the supplied dataset.
    ValueError
        DataFrame supplied, but no features.
    ValueError
        target_feature not found in the dataset.
    TypeError
        Supplied dataset was not of type Phenonaut, Dataset, or pd.DataFrame
    """

    import pandas as pd

    # If Phenonaut object supplied, change it to the dataset
    if isinstance(dataset, Phenonaut):
        if len(dataset.datasets) != 1:
            raise ValueError(
                f"Phenonaut object passed to feature_correlation_to_target but it did not contain 1 dataset, it contained {len(dataset.datasets)}"
            )
        dataset = dataset[-1]

    # If Dataset, then make df with just required features
    if isinstance(dataset, Dataset):
        if features is None:
            features = list(dataset.features)
        if target_feature in features:
            features.remove(target_feature)
        if target_feature not in dataset.df.columns:
            raise ValueError(
                "Given target_feature was not found in the supplied Dataset"
            )
        dataset = dataset.df[features + [target_feature]]

    elif isinstance(dataset, DataFrame):
        if features is None:
            raise ValueError(
                "DataFrame provided to feature_correlation_to_target, but no features_list"
            )
        if target_feature not in dataset.columns:
            raise ValueError(
                "Given target_feature was not found in the supplied Dataset"
            )
        dataset = dataset[features + [target_feature]]
    else:
        raise TypeError(
            "dataset argument to feature_correlation_to_target was not of type Phenonaut, Dataset, or DataFrame"
        )

    coefs = {
        column_name: dataset[column_name].corr(dataset[target_feature], method=method)
        for column_name in features
    }

    if return_dataframe:
        return (
            DataFrame.from_dict(
                {
                    "index": coefs.keys(),
                    f"correlation_to_{target_feature}": coefs.values(),
                }
            )
            .set_index("index")
            .sort_values(by=f"correlation_to_{target_feature}", ascending=False)
        )
    else:
        return coefs
