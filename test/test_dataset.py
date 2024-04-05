# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import numpy as np
import pandas as pd

import phenonaut
from phenonaut.data.dataset import Dataset
from phenonaut.phenonaut import Phenonaut


def test_get_features_and_non_features(dataset_iris):
    features = dataset_iris.features
    print(features)
    non_features = dataset_iris.get_non_feature_columns()
    assert sorted(features + non_features) == sorted(dataset_iris.df.columns.values)


def test_new_aggregated_dataset(small_2_plate_df):
    """Here we test a df as follows:
    ROW	COLUMN	BARCODE	feat_1	feat_2	feat_3	filename	FOV
    1	1	    Plate1	1.2	    1.2	    1.3	    fileA.png	1
    1	1	    Plate1	1.3	    1.4	    1.5	    FileB.png	2
    1	1	    Plate2	5.2	    5.1	    5	    FileC.png	1
    1	1	    Plate2	6.2	    6.1	    6.8	    FileD.png	2
    1	2	    Plate1	0.1	    0.2	    0.3	    fileE.png	1
    1	2	    Plate1	0.2	    0.2	    0.38    FileF.png	2

    As there are multiple fields of view, we merge on common ROW, COLUMN and BARCODE fields.
    Merging uses np.mean, but may be any callable which returns a sing value

    """
    phe = phenonaut.Phenonaut(small_2_plate_df)

    new_ds = phe.ds.new_aggregated_dataset(["ROW", "COLUMN", "BARCODE"])
    assert set(new_ds.features) == {"feat_1", "feat_2", "feat_3"}
    assert abs(new_ds.df["feat_1"][0] - 1.25) < 1e-6
    assert np.abs(new_ds.df["feat_1"][1] - 5.70) < 1e-6
    assert np.abs(new_ds.df["feat_1"][2] - 0.15) < 1e-6
    assert np.abs(new_ds.df["feat_2"][0] - 1.3) < 1e-6
    assert np.abs(new_ds.df["feat_2"][1] - 5.6) < 1e-6
    assert np.abs(new_ds.df["feat_2"][2] - 0.2) < 1e-6
    assert np.abs(new_ds.df["feat_3"][0] - 1.4) < 1e-6
    assert np.abs(new_ds.df["feat_3"][1] - 5.9) < 1e-6
    assert np.abs(new_ds.df["feat_3"][2] - 0.34) < 1e-6


def test_iris_packageddataset():
    import tempfile

    tmpdir = tempfile.gettempdir()
    from phenonaut.packaged_datasets import Iris, Iris_2_views

    phe = Phenonaut(Iris(tmpdir))
    assert len(phe.keys()) == 1


def test_feature_selection_when_loading_csv():
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            "row, column, feature1, feature2, derived_feature3, feature10, feature003\n1,1,1,2,3,4,5\n2,1,6,7,8,9,10"
        )
        tmp.flush()
        phe = Phenonaut()
        phe.load_dataset(tmp.name, tmp.name, {"features_prefix": "feat"})
        assert phe.ds.features == ["feature1", "feature2", "feature003", "feature10"]

    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target\n5.4,3.4,1.7,0.2,0\n7.2,3.0,5.8,1.6,2\n6.4,2.8,5.6,2.1,2\n4.8,3.1,1.60.2,0\n5.6,2.5,3.9,1.1,1"
        )
        tmp.flush()
        phe = Phenonaut()
        phe.load_dataset(tmp.name, tmp.name, {"features_regex": ".*(width|length).*"})
        assert phe.ds.features == [
            "petal length (cm)",
            "petal width (cm)",
            "sepal length (cm)",
            "sepal width (cm)",
        ]


def test_replace_string_in_column(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    phe[-1].replace_str("BARCODE", "Plate", "P")
    assert all(elem in ["P1", "P2"] for elem in phe.df.BARCODE.unique())


def test_remove_features_with_outliers(dataset_iris):
    dataset_iris.remove_features_with_outliers(7.5)
    assert len(dataset_iris.features) == 3


def test_remove_low_variance_features(dataset_iris):
    dataset_iris.remove_low_variance_features(freq_cutoff=0.5, unique_cutoff=0.2)
    assert dataset_iris.features == ["petal length (cm)", "sepal length (cm)"]


def test_remove_blocklist_features(dataset_iris):
    dataset_iris.remove_blocklist_features(["petal length (cm)", "sepal length (cm)"])
    assert dataset_iris.features == ["petal width (cm)", "sepal width (cm)"]

    dataset_iris.rename_column(
        "petal width (cm)", "RobustMAD_Nuclei_Correlation_Manders_AGP_DNA"
    )

    dataset_iris.remove_blocklist_features("CellProfiler")
    assert "RobustMAD_Nuclei_Correlation_Manders_AGP_DNA" not in dataset_iris.df.columns
    assert dataset_iris.features == ["sepal width (cm)"]


def test_drop_nans_with_cutoff(nan_inf_dataset):
    nan_inf_dataset.drop_nans_with_cutoff(nan_cutoff=0.4)
    assert list(nan_inf_dataset.df.columns) == ["A", "B", "C", "D", "F"]
    assert list(nan_inf_dataset.df.index) == [0, 1, 2, 5]
    assert nan_inf_dataset.df.equals(
        pd.DataFrame(
            {
                "A": {0: 1, 1: 2, 2: 3, 5: 6},
                "B": {0: 6, 1: 5, 2: 4, 5: 1},
                "C": {0: 1.0, 1: 2.0, 2: 3.0, 5: 4.0},
                "D": {0: np.inf, 1: 1.0, 2: 2.0, 5: np.nan},
                "F": {0: "g1", 1: "g1", 2: "g1", 5: "g2"},
            }
        )
    )


def test_impute_nans(nan_inf_dataset):
    nan_inf_dataset.impute_nans(groupby_col=None, impute_fn="median")

    assert nan_inf_dataset.df.equals(
        pd.DataFrame(
            {
                "A": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
                "B": {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1},
                "C": {0: 1.0, 1: 2.0, 2: 3.0, 3: 2.5, 4: 2.5, 5: 4.0},
                "D": {0: 2.5, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 2.5},
                "E": {0: 5.0, 1: 5.5, 2: 5.5, 3: 5.5, 4: 5.5, 5: 6.0},
                "F": {0: "g1", 1: "g1", 2: "g1", 3: "g2", 4: "g2", 5: "g2"},
            }
        )
    )


def test_subtract_from_datasets(small_2_plate_ds):
    ds = small_2_plate_ds
    ds.subtract_median(query_or_perturbation_name="FOV==1", groupby="BARCODE")
    assert (
        np.abs(
            np.sum(ds.df["feat_1"] - np.array([0.55, 0.65, -0.55, -0.45, 0.00, 1.00]))
        )
        < 1e-6
    )


def test_dataset_groupby(small_2_plate_df):
    """Test split of a Dataset comprising the data below, by filename and FOV
    ROW	COLUMN	BARCODE	feat_1	feat_2	feat_3	filename	FOV
    1	1	    Plate1	1.2	    1.2	    1.3	    fileA.png	1
    1	1	    Plate1	1.3	    1.4	    1.5	    FileB.png	2
    1	1	    Plate2	5.2	    5.1	    5	    FileC.png	1
    1	1	    Plate2	6.2	    6.1	    6.8	    FileD.png	2
    1	2	    Plate1	0.1	    0.2	    0.3	    fileE.png	1
    1	2	    Plate1	0.2	    0.2	    0.38    FileF.png	2

    """
    phe = phenonaut.Phenonaut(
        small_2_plate_df, metadata={"features_prefix": "feat_"}
    )
    new_ds = phe.ds.new_aggregated_dataset(["ROW", "COLUMN", "BARCODE"])
    split_ds = phe.ds.groupby(["FOV", "filename"])
    assert len(split_ds) == 6
    assert split_ds[0].df.FOV.unique()[0] == 1
    assert split_ds[-1].df.FOV.unique()[0] == 2
