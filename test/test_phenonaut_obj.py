# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from curses import meta
import pytest
import phenonaut
from phenonaut.data.dataset import Dataset
import pandas as pd
import numpy as np
import tempfile
from io import StringIO


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
    all_possible_views = phenonaut_object_iris_4_views.get_dataset_combinations(return_indexes=True)
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
    assert set(ph0.ds.features) == set(["B", "A", "id"])
    ph0 = phenonaut.Phenonaut([df, another_df], metadata={"features_prefix": ""})
    assert set(ph0.datasets[0].features) == set(["B", "A", "id"])
    assert set(ph0.datasets[1].features) == set(["B", "A2", "id2"])
    ph0 = phenonaut.Phenonaut([df, another_df], metadata={"features": ["B"]})
    assert set(ph0.datasets[0].features) == set(["B"])
    assert set(ph0.datasets[1].features) == set(["B"])
    ph0 = phenonaut.Phenonaut(
        [df, another_df],
        metadata=[{"features": ["B"]}, {"features_prefix": ""}],
        dataframe_name=["Bob", "Robert"],
    )
    assert set(ph0.datasets[0].features) == set(["B"])
    assert set(ph0.datasets[1].features) == set(["A2", "B", "id2"])
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
    assert set(phe.ds.features) == set(["feat_1", "feat_2", "feat_3"])
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
    assert set(phe.ds.features) == set(["feat_1", "feat_2", "feat_3"])
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


def test_phe_save_and_load(dataset_iris):
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    try:
        phe = Phenonaut(dataset_iris)
        phe.save(tmp_file.name)
        phe_loaded = Phenonaut.load(tmp_file.name)
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


def test_readin_transformations():
    df = pd.read_csv(
        StringIO(
            """"idx","Var1","Var2","value"
        
        "1","ENSG00000225972","A1_CPD1_PLATE1",4.0185
        "2","ENSG00000225630","A1_CPD1_PLATE1",1.1539
        "3","ENSG00000225972","A2_CPD2_PLATE1",10.6661
        "4","ENSG00000225630","A2_CPD2_PLATE1",1.6130
        "5","ENSG00000225972","A3_CPD3_PLATE1",0.1234
        "6","ENSG00000225630","A3_CPD3_PLATE1",9.8436
        "7","ENSG00000225972","A4_Pos_ctrl_PLATE1",0.1234
        "8","ENSG00000225630","A4_Pos_ctrl_PLATE1",9.8436"""
        )
    ).reset_index()
    phe = Phenonaut(
        df,
        "Test_df",
        metadata={
            "transforms": [
                ("replace_str", ("Var2", "Pos_", "Pos-")),
                ("split_column", ("Var2", "_", ["Well", "CpdID", "PlateID"])),
                ("pivot", ("Var1", "value")),
                #'transpose',
                # (pivot_table(values = 'value', index=["WellName"], columns = 'Var1')
            ],
            "features_prefix": "ENSG",
        },
    )
    print(phe.df)
