import numpy as np
import pytest
import phenonaut
from phenonaut.metrics.performance import mp_value_score
from phenonaut.errors import NotEnoughRowsError
import pandas as pd


def test_mp_value_score(twenty_one_blobs_phe):
    """Test a selection of phenotypic metrics"""
    phe = twenty_one_blobs_phe
    results = mp_value_score(
        phe,
        "label",
        "label==0",
    )
    print(results)
    assert results.loc[0, "mp_value"] == 1.0
    assert all(results.loc[1:, "mp_value"] == 0.0)

    # Check a helpful error is thown when an invalid query is supplied
    with pytest.raises(NotEnoughRowsError):
        _ = mp_value_score(phe, "label", "label==-1")

    # Check a helpful error is thrown when a grouped DF group contains < 3 rows
    tmp_ds = phe.ds.copy()
    tmp_ds.df = pd.concat(
        [tmp_ds.df.query("label!=0"), tmp_ds.df.query("label==0").sample(1)]
    )
    with pytest.raises(NotEnoughRowsError):
        _ = mp_value_score(tmp_ds, "label", "label==1")

    tmp_ds = phe.ds.copy()
    tmp_ds.df = pd.concat(
        [tmp_ds.df.query("label!=0"), tmp_ds.df.query("label==0").sample(2)]
    )
    with pytest.raises(NotEnoughRowsError):
        _ = mp_value_score(tmp_ds, "label", "label==1")

    # And check all ok with 3 rows
    tmp_ds = phe.ds.copy()
    tmp_ds.df = pd.concat(
        [tmp_ds.df.query("label!=0"), tmp_ds.df.query("label==0").sample(3)]
    )
    _ = mp_value_score(tmp_ds, "label", "label==1")
