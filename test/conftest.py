# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import phenonaut
from phenonaut.data import Dataset


@pytest.fixture
def sklearn_regression_df():
    """Return sklearn regression dataframe generated by sklearn.datasets.make_regression

    sklearn generated regression datasets, 100 rows, 100 features (named feat_n where n
    is 1-100), and one target column which is a regression target.

    Generated using the following python code:
        from sklearn.datasets import make_regression
        X,y=make_regression(random_state=42)
        df = pd.DataFrame(np.hstack([X,y.reshape(-1,1)]), columns=[f"feat_{i+1}" for i in range(X.shape[1])]+["target"])
        df.to_csv("test/generated_regression_dataset.csv")

    """
    return pd.read_csv(Path("test") / "generated_regression_dataset.csv", index_col=0)


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
def small_2_plate_ds(small_2_plate_df: pd.DataFrame):
    return Dataset(
        "Small 2 plate DS",
        small_2_plate_df,
        {"features": ["feat_1", "feat_2", "feat_3"]},
    )


@pytest.fixture
def iris_df():
    df = load_iris(as_frame=True).frame
    return df


@pytest.fixture
def dataset_iris(iris_df: pd.DataFrame):
    column_names = iris_df.columns.to_list()
    return Dataset("Iris", iris_df, {"features": column_names[0:4]})


@pytest.fixture
def phenonaut_object_iris_2_views(iris_df: pd.DataFrame):
    column_names = iris_df.columns.to_list()
    df1 = iris_df.iloc[:, [0, 1, 4]].copy()
    df2 = iris_df.iloc[:, [2, 3, 4]].copy()
    ds1 = Dataset("Iris_view1", df1, {"features": column_names[0:2]})
    ds2 = Dataset("Iris_view2", df2, {"features": column_names[2:4]})
    return phenonaut.Phenonaut([ds1, ds2])


@pytest.fixture
def phenonaut_object_iris_4_views(iris_df: pd.DataFrame):
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
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6],
            "B": [6, 5, 4, 3, 2, 1],
            "C": [1, 2, 3, np.nan, np.inf, 4],
            "D": [np.inf, 1, 2, 3, 4, np.nan],
            "E": [5, np.nan, np.nan, np.nan, np.nan, 6],
            "F": ["g1", "g1", "g1", "g2", "g2", "g2"],
        }
    )

    return df


@pytest.fixture
def nan_inf_dataset(nan_inf_df: pd.DataFrame):
    column_names = nan_inf_df.columns.to_list()
    return Dataset("nan_inf", nan_inf_df, {"features": column_names[0:5]})


@pytest.fixture
def twenty_one_blobs_phe():
    """Make a 20-feature dataset containing 20 5x replicates and 1 20x negative control

    Last compound has a large SD

    Returns
    -------
    phenonaut.Phenonaut
        Phenonaut object containing blobs
    """
    import phenonaut
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_blobs

    num_features = 10
    num_treatments = 20
    num_vehicle_replicates = 20
    num_treatment_replicates = 5

    X, y = make_blobs(
        n_samples=[num_vehicle_replicates]
        + [num_treatment_replicates] * num_treatments,
        n_features=num_features,
        cluster_std=[2] + [1] * (num_treatments - 1) + [4],
        random_state=42,
        return_centers=False,
        shuffle=False,
    )
    df = pd.DataFrame(
        np.hstack([X, y[:, None]]),
        columns=[f"feat_{f+1}" for f in range(num_features)] + ["label"],
    )
    df["label"] = df["label"].astype(int)
    phe = phenonaut.Phenonaut(df, metadata={"features_prefix": "feat_"})
    phe.ds.perturbation_column = "label"
    return phe



@pytest.fixture
def synthetic_screening_dataset_1():
    """Return simulated screening dataset #1

    sklearn generated regression datasets, 100 rows, 100 features (named feat_n where n
    is 1-100), and one target column which is a regression target.

    Generated using the following python code:
        from sklearn.datasets import make_regression
        X,y=make_regression(random_state=42)
        df = pd.DataFrame(np.hstack([X,y.reshape(-1,1)]), columns=[f"feat_{i+1}" for i in range(X.shape[1])]+["target"])
        df.to_csv("test/generated_regression_dataset.csv")

    """
    n_dims = 3
    n_dmso_wells = 64
    n_treatments = 5
    n_replicates = 4
    random_state = np.random.default_rng(7)
    dataframes = []

    # Append the DMSO DF
    dataframes.append(
        pd.concat(
            [
                pd.DataFrame(
                    random_state.normal([0] * n_dims, 1, size=(n_dmso_wells, n_dims)),
                    columns=[f"feat_{i+1}" for i in range(n_dims)],
                ),
                pd.Series(["DMSO"] * n_dmso_wells, name="pert_iname"),
            ],
            axis=1,
        )
    )

    # Append treatment DFs
    for replicate_i in range(n_treatments):
        dataframes.append(
            pd.concat(
                [
                    pd.DataFrame(
                        random_state.normal(
                            random_state.uniform(-10, 10, n_dims),
                            random_state.uniform(0.5, 3),
                            size=(n_replicates, n_dims),
                        ),
                        columns=[f"feat_{i+1}" for i in range(n_dims)],
                    ),
                    pd.Series(
                        [f"Trt_{replicate_i+1}"] * n_replicates, name="pert_iname"
                    ),
                ],
                axis=1,
            )
        )

    return pd.concat(dataframes)



@pytest.fixture
def synthetic_screening_dataset_2():
    """Return simulated screening dataset #1

    sklearn generated regression datasets, 100 rows, 100 features (named feat_n where n
    is 1-100), and one target column which is a regression target.

    Generated using the following python code:
        from sklearn.datasets import make_regression
        X,y=make_regression(random_state=42)
        df = pd.DataFrame(np.hstack([X,y.reshape(-1,1)]), columns=[f"feat_{i+1}" for i in range(X.shape[1])]+["target"])
        df.to_csv("test/generated_regression_dataset.csv")

    """
    n_dims = 3
    n_dmso_wells = 64
    n_treatments = 10
    n_replicates = 5
    random_state = np.random.default_rng(7)
    dataframes = []

    # Append the DMSO DF
    dataframes.append(
        pd.concat(
            [
                pd.DataFrame(
                    random_state.normal([0] * n_dims, 1, size=(n_dmso_wells, n_dims)),
                    columns=[f"feat_{i+1}" for i in range(n_dims)],
                ),
                pd.Series(["DMSO"] * n_dmso_wells, name="pert_iname"),
            ],
            axis=1,
        )
    )

    # Append treatment DFs
    for replicate_i in range(n_treatments):
        dataframes.append(
            pd.concat(
                [
                    pd.DataFrame(
                        random_state.normal(
                            random_state.uniform(-1, 1, n_dims),
                            random_state.uniform(0.5, 2),
                            size=(n_replicates, n_dims),
                        ),
                        columns=[f"feat_{i+1}" for i in range(n_dims)],
                    ),
                    pd.Series(
                        [f"Trt_{replicate_i+1}"] * n_replicates, name="pert_iname"
                    ),
                ],
                axis=1,
            )
        )

    return pd.concat(dataframes)