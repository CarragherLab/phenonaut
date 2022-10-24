# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.transforms.preparative import VIF, RemoveHighlyCorrelated, RemoveHighestCorrelatedThenVIF
import pandas as pd
import pytest
from pathlib import Path

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
    return pd.read_csv(Path("test")/"generated_regression_dataset.csv", index_col=0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_VIF(sklearn_regression_df):
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    vif=VIF()
    vif.filter(phe.ds, vif_cutoff=2, min_features=50)
    assert len(phe.ds.features)==52
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    vif=VIF()
    vif(phe.ds, vif_cutoff=2, min_features=60)
    assert len(phe.ds.features)==79

def test_RemoveHighlyCorrelated(sklearn_regression_df):
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    rhc=RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=0.3)
    assert len(phe.ds.features)==95
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    rhc=RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=0.4, min_features=97)
    assert len(phe.ds.features)==97
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    rhc=RemoveHighlyCorrelated()
    rhc(phe.ds, threshold=None, min_features=5)
    assert len(phe.ds.features)==5

def test_RemoveHighestCorrelatedThenVIF(sklearn_regression_df):
    phe=Phenonaut(sklearn_regression_df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    rhcvif=RemoveHighestCorrelatedThenVIF()
    rhcvif(phe.ds, n_before_vif=95, vif_cutoff=5.0)
    assert len(phe.ds.features)==78