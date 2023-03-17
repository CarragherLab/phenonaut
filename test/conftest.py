# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import phenonaut
import pytest
from phenonaut.data import Dataset
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


@pytest.fixture
def small_2_plate_df():
    return pd.DataFrame(
        {
            "ROW": [1, 1, 1, 1, 1, 1],
            "COLUMN": [1, 1, 1, 1, 2, 2],
            "BARCODE": ["Plate1", "Plate1", "Plate2", "Plate2", "Plate1", "Plate1"],
            "feat_1": [1.2, 1.3, 5.2, 6.2, 0.1, 0.2],
            "feat_2": [1.2, 1.4, 5.1, 6.1, 0.2, 0.2],
            "feat_3": [1.3, 1.5, 5, 6.8, 0.3, 0.38],
            "filename": [
                "fileA.png",
                "FileB.png",
                "FileC.png",
                "FileD.png",
                "fileE.png",
                "FileF.png",
            ],
            "FOV": [1, 2, 1, 2, 1, 2],
        }
    )

@pytest.fixture
def small_2_plate_ds(small_2_plate_df:pd.DataFrame):
    ds=Dataset("Small 2 plate DS", small_2_plate_df, {"features": ['feat_1','feat_2', 'feat_3']})
    return Dataset("Small 2 plate DS", small_2_plate_df, {"features": ['feat_1','feat_2', 'feat_3']})


@pytest.fixture
def iris_df():
    df = load_iris(as_frame=True).frame
    return df


@pytest.fixture
def dataset_iris(iris_df:pd.DataFrame):
    column_names = iris_df.columns.to_list()
    return Dataset("Iris", iris_df, {"features": column_names[0:4]})


@pytest.fixture
def phenonaut_object_iris_2_views(iris_df:pd.DataFrame):
    column_names = iris_df.columns.to_list()
    df1 = iris_df.iloc[:, [0, 1, 4]].copy()
    df2 = iris_df.iloc[:, [2, 3, 4]].copy()
    ds1 = Dataset("Iris_view1", df1, {"features": column_names[0:2]})
    ds2 = Dataset("Iris_view2", df2, {"features": column_names[2:4]})
    return phenonaut.Phenonaut([ds1, ds2])


@pytest.fixture
def phenonaut_object_iris_4_views(iris_df:pd.DataFrame):
    column_names = iris_df.columns.to_list()
    df1 = iris_df.iloc[:, [0, 4]].copy()
    df2 = iris_df.iloc[:, [1, 4]].copy()
    df3 = iris_df.iloc[:, [2, 4]].copy()
    df4 = iris_df.iloc[:, [3, 4]].copy()
    ds1 = Dataset("Iris_view1", df1, {"features": [column_names[0]]})
    ds2 = Dataset("Iris_view2", df2, {"features": [column_names[1]]})
    ds3 = Dataset("Iris_view3", df3, {"features": [column_names[2]]})
    ds4 = Dataset("Iris_view4", df4, {"features": [column_names[3]]})
    return phenonaut.Phenonaut([ds1, ds2, ds3, ds4])

@pytest.fixture
def nan_inf_df():

    df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6],
            'B': [6, 5, 4, 3, 2, 1],
            'C': [1, 2, 3, np.nan, np.inf, 4],
            'D': [np.inf, 1, 2, 3, 4, np.nan],
            'E': [5, np.nan, np.nan, np.nan, np.nan, 6],
            'F': ["g1","g1","g1","g2","g2","g2"]
            })

    return df

@pytest.fixture
def nan_inf_dataset(nan_inf_df: pd.DataFrame):
    column_names = nan_inf_df.columns.to_list()
    return Dataset("nan_inf", nan_inf_df , {"features": column_names[0:5]})