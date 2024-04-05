# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.transforms.preparative import (
    VIF,
    RemoveHighestCorrelatedThenVIF,
    RemoveHighlyCorrelated,
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_VIF(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    vif = VIF()
    vif.filter(phe.ds, vif_cutoff=2, min_features=50)
    assert len(phe.ds.features) == 52
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    vif = VIF()
    vif(phe.ds, vif_cutoff=2, min_features=60)
    assert len(phe.ds.features) == 79


def test_RemoveHighlyCorrelated(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    rhc = RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=0.3)
    assert len(phe.ds.features) == 95
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    rhc = RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=0.4, min_features=97)
    assert len(phe.ds.features) == 97
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    rhc = RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=None, min_features=5)
    assert len(phe.ds.features) == 5


def test_RemoveHighestCorrelatedThenVIF(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    rhcvif = RemoveHighestCorrelatedThenVIF()
    rhcvif(phe.ds, n_before_vif=95, vif_cutoff=5.0)
    assert len(phe.ds.features) == 78


def test_SimpleImputer(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    df = phe.ds.df
    df.loc[1, phe.ds.features] = np.nan
    print(phe.df[phe.ds.features].head())
    from phenonaut.transforms.imputers import SimpleImputer

    imputer = SimpleImputer()
    imputer(phe)
    assert not phe.ds.data.isnull().values.any()


def test_KNNImputer(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    df = phe.ds.df
    df.loc[1, phe.ds.features] = np.nan
    print(phe.df[phe.ds.features].head())
    from phenonaut.transforms.imputers import KNNImputer

    imputer = KNNImputer()
    imputer(phe)
    assert not phe.ds.data.isnull().values.any()


def test_RFImputer(sklearn_regression_df):
    phe = Phenonaut(
        sklearn_regression_df, "PheSmallDS", metadata={"features_prefix": "feat_"}
    )
    df = phe.ds.df
    df.loc[1, phe.ds.features] = np.nan
    print(phe.df[phe.ds.features].head())
    from phenonaut.transforms.imputers import RFImputer

    imputer = RFImputer(max_iter=3)
    imputer(phe)
    print(phe.df[phe.ds.features].head())
    assert not phe.ds.data.isnull().values.any()
