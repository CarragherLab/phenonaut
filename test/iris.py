# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import phenonaut
import pytest
from phenonaut.data import Dataset
from sklearn.datasets import load_iris


@pytest.fixture
def iris_df():
    df = load_iris(as_frame=True).frame
    return df


@pytest.fixture
def dataset_iris(iris_df):
    column_names = iris_df.columns.to_list()
    return Dataset("Iris", iris_df, {"features": column_names[0:4]})


@pytest.fixture
def phenonaut_object_iris_2_views(iris_df):
    column_names = iris_df.columns.to_list()
    df1 = iris_df.iloc[:, [0, 1, 4]].copy()
    df2 = iris_df.iloc[:, [2, 3, 4]].copy()
    ds1 = Dataset("Iris_view1", df1, {"features": column_names[0:2]})
    ds2 = Dataset("Iris_view2", df2, {"features": column_names[2:4]})
    return phenonaut.Phenonaut([ds1, ds2])


@pytest.fixture
def phenonaut_object_iris_4_views(iris_df):
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
