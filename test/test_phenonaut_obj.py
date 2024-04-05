# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import tempfile
from curses import meta
from io import StringIO

import numpy as np
import pandas as pd
import pytest

import phenonaut
from phenonaut.data.dataset import Dataset
from phenonaut.phenonaut import Phenonaut


def test_possible_dataset_combinations(phenonaut_object_iris_4_views):
    all_possible_views = phenonaut_object_iris_4_views.get_dataset_combinations()
    assert all_possible_views == (
        ("Iris_view1",),
        ("Iris_view2",),
        ("Iris_view3",),
        ("Iris_view4",),
        ("Iris_view1", "Iris_view2"),
        ("Iris_view1", "Iris_view3"),
        ("Iris_view1", "Iris_view4"),
        ("Iris_view2", "Iris_view3"),
        ("Iris_view2", "Iris_view4"),
        ("Iris_view3", "Iris_view4"),
        ("Iris_view1", "Iris_view2", "Iris_view3"),
        ("Iris_view1", "Iris_view2", "Iris_view4"),
        ("Iris_view1", "Iris_view3", "Iris_view4"),
        ("Iris_view2", "Iris_view3", "Iris_view4"),
        ("Iris_view1", "Iris_view2", "Iris_view3", "Iris_view4"),
    )


def test_possible_dataset_combinations_indexes(phenonaut_object_iris_4_views):
    all_possible_views = phenonaut_object_iris_4_views.get_dataset_combinations(
        return_indexes=True
    )
    assert all_possible_views == (
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    )


def test_get_dataset_index_from_name(phenonaut_object_iris_4_views):
    assert phenonaut_object_iris_4_views.get_dataset_index_from_name(
        ["Iris_view4", "Iris_view3", "Iris_view2", "Iris_view1"]
    ) == [3, 2, 1, 0]


def test_phenonaut_obj_from_df():
    df = pd.DataFrame({"id": [9, 8, 7], "A": [1, 2, 3], "B": [4, 5, 6]})
    another_df = pd.DataFrame({"id2": [9, 8, 7], "A2": [1, 2, 3], "B": [4, 5, 6]})
    ph0 = phenonaut.Phenonaut(df, metadata={"features": ["B"]}, dataframe_name="Bob")
    assert ph0.ds.features == ["B"]
    assert ph0.ds.name == "Bob"
    ph0 = phenonaut.Phenonaut(df, metadata={"features_prefix": ["B"]})
    assert ph0.ds.features == ["B"]
    ph0 = phenonaut.Phenonaut(df, metadata={"features_prefix": ""})
    assert set(ph0.ds.features) == {"B", "A", "id"}
    ph0 = phenonaut.Phenonaut([df, another_df], metadata={"features_prefix": ""})
    assert set(ph0.datasets[0].features) == {"B", "A", "id"}
    assert set(ph0.datasets[1].features) == {"B", "A2", "id2"}
    ph0 = phenonaut.Phenonaut([df, another_df], metadata={"features": ["B"]})
    assert set(ph0.datasets[0].features) == {"B"}
    assert set(ph0.datasets[1].features) == {"B"}
    ph0 = phenonaut.Phenonaut(
        [df, another_df],
        metadata=[{"features": ["B"]}, {"features_prefix": ""}],
        dataframe_name=["Bob", "Robert"],
    )
    assert set(ph0.datasets[0].features) == {"B"}
    assert set(ph0.datasets[1].features) == {"A2", "B", "id2"}
    assert ph0.datasets[0].name == "Bob"
    assert ph0.datasets[-1].name == "Robert"


def test_aggregate_dataset_in_ph0_obj():
    """Here we test a df as follows:
    ROW	COLUMN	BARCODE	feat_1	feat_2	feat_3	filename	FOV
    1	1	    Plate1	1.2	    1.2	    1.3	    fileA.png	1
    1	1	    Plate1	1.3	    1.4	    1.5	    FileB.png	2
    1	1	    Plate2	5.2	    5.1	    5	    FileC.png	1
    1	1	    Plate2	6.2	    6.1	    6.8	    FileD.png	2
    1	2	    Plate1	0.1	    0.2	    0.3	    fileE.png	1
    1	2	    Plate1	0.2	    0.2	    0.38    FileF.png	2

    As there are multiple fields of view, we merge on common ROW, COLUMN and BARCODE fields.
    Merging uses np.mean, but may be any callable which returns a single value

    """

    df = pd.DataFrame(
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

    phe = phenonaut.Phenonaut(df, metadata={"features_prefix": "feat_"})
    phe.aggregate_dataset(["ROW", "COLUMN", "BARCODE"])
    assert len(phe.datasets) == 2
    assert set(phe.ds.features) == {"feat_1", "feat_2", "feat_3"}
    assert abs(phe.ds.df["feat_1"][0] - 1.25 < 0.00001)
    assert np.abs(phe.ds.df["feat_1"][1] - 5.70 < 0.00001)
    assert np.abs(phe.ds.df["feat_1"][2] - 0.15 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][0] - 1.3 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][1] - 5.6 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][2] - 0.2 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][0] - 1.4 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][1] - 5.9 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][2] - 0.34 < 0.00001)


def test_aggregate_dataset():
    """Here we test a df as follows:
    ROW	COLUMN	BARCODE	feat_1	feat_2	feat_3	filename	FOV
    1	1	    Plate1	1.2	    1.2	    1.3	    fileA.png	1
    1	1	    Plate1	1.3	    1.4	    1.5	    FileB.png	2
    1	1	    Plate2	5.2	    5.1	    5	    FileC.png	1
    1	1	    Plate2	6.2	    6.1	    6.8	    FileD.png	2
    1	2	    Plate1	0.1	    0.2	    0.3	    fileE.png	1
    1	2	    Plate1	0.2	    0.2	    0.38    FileF.png	2

    As there are multiple fields of view, we merge on common ROW, COLUMN and BARCODE fields.
    Merging uses np.mean, but may be any callable which returns a single value

    """
    df = pd.DataFrame(
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

    phe = phenonaut.Phenonaut([df, df])
    phe.aggregate_dataset(
        ["ROW", "COLUMN", "BARCODE"], datasets=[0, 1], new_names_or_prefix=["M1", "M2"]
    )
    assert len(phe.datasets) == 4
    phe = phenonaut.Phenonaut([df, df])
    phe.aggregate_dataset(
        ["ROW", "COLUMN", "BARCODE"],
        datasets=[0, 1],
        new_names_or_prefix=["M1", "M2"],
        inplace=True,
    )
    assert len(phe.datasets) == 2
    assert set(phe.ds.features) == {"feat_1", "feat_2", "feat_3"}
    assert abs(phe.ds.df["feat_1"][0] - 1.25 < 0.00001)
    assert np.abs(phe.ds.df["feat_1"][1] - 5.70 < 0.00001)
    assert np.abs(phe.ds.df["feat_1"][2] - 0.15 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][0] - 1.3 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][1] - 5.6 < 0.00001)
    assert np.abs(phe.ds.df["feat_2"][2] - 0.2 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][0] - 1.4 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][1] - 5.9 < 0.00001)
    assert np.abs(phe.ds.df["feat_3"][2] - 0.34 < 0.00001)


def test_phe_getitem_setitem_delitem(dataset_iris):
    phe = Phenonaut(dataset_iris)
    phe["test_ds"] = dataset_iris.copy()
    assert len(phe.datasets) == 2
    assert len(phe[[0, 1]]) == 2
    assert phe["test_ds"].data.shape[1] == 4
    del phe["Iris"]
    assert len(phe.datasets) == 1
    assert phe.datasets[0].name == "test_ds"


def test_phe_save_load_and_revert(dataset_iris):
    tmp_file = tempfile.NamedTemporaryFile(delete=True)

    try:
        phe = Phenonaut(dataset_iris)
        # Save, using overwrite_existing=True, as NamedTemporaryFile makes the
        # file, Phenonaut would then throw a warning saying it exists.
        phe.save(tmp_file.name, overwrite_existing=True)

        phe_loaded = Phenonaut.load(tmp_file.name)
        assert len(phe_loaded.datasets) == 1
        phe.revert()
        assert len(phe_loaded.datasets) == 1
    finally:
        tmp_file.close()


def test_phe_clone(dataset_iris):
    phe = Phenonaut(dataset_iris)
    phe.clone_dataset("Iris", "Iris2")
    phe.clone_dataset("Iris", "Iris2", overwrite_existing=True)
    phe.clone_dataset("Iris", "Iris3", overwrite_existing=True)
    phe.clone_dataset("Iris", "Iris4", overwrite_existing=False)
    assert phe.keys() == ["Iris", "Iris2", "Iris3", "Iris4"]


def test_splitting_dataset_in_ph0_obj(small_2_plate_df):
    """Test the splitting of datasets using groupby"""
    phe = phenonaut.Phenonaut(
        small_2_plate_df, metadata={"features_prefix": "feat_"}
    )
    assert len(phe.datasets) == 1
    phe.groupby_datasets(["FOV", "filename"])
    assert len(phe.datasets) == 6

    phe = phenonaut.Phenonaut(
        small_2_plate_df, metadata={"features_prefix": "feat_"}
    )
    assert len(phe.datasets) == 1
    phe.groupby_datasets(["FOV", "filename"], remove_original=False)
    assert len(phe.datasets) == 7


def test_merge_datasets(small_2_plate_df):
    """Test merging of datasets split using the groupby method"""
    phe = phenonaut.Phenonaut(
        small_2_plate_df, metadata={"features_prefix": "feat_"}
    )
    split_ds = phe.ds.groupby(["FOV", "filename"])
    phe.merge_datasets(split_ds)
    assert len(phe.ds.df) == len(small_2_plate_df)
    del phe[-1]
    merged_ds = phe.merge_datasets(split_ds, return_merged=True)
    assert len(merged_ds.df) == len(small_2_plate_df)
    phe.datasets.extend(split_ds)
    del phe[0]
    phe.merge_datasets(list(range(0, len(split_ds))))
    assert len(phe.datasets) == 1
