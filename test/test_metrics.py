import pytest
import phenonaut
from phenonaut.metrics import mp_value_score
from phenonaut.errors import NotEnoughRowsError
from phenonaut.metrics import (
    percent_replicating,
    pertmutation_test_distinct_from_query_group,
    auroc,
)
from phenonaut.metrics.non_ds_phenotypic_metrics import non_ds_phenotypic_metrics
import numpy as np
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


def test_compactness_percent_replicating(synthetic_screening_dataset_1):
    pr_results = percent_replicating(
        synthetic_screening_dataset_1,
        "pert_iname",
        "pert_iname!='DMSO'",
        features=["feat_1", "feat_2", "feat_3"],
    )
    assert pr_results == pytest.approx(80.0)


def test_distinctness_permutation_test_against_DMSO_quick(
    synthetic_screening_dataset_1,
):
    euclidean_metric = non_ds_phenotypic_metrics["Euclidean"]
    phe = phenonaut.Phenonaut(
        synthetic_screening_dataset_1,
        metadata={"features": ["feat_1", "feat_2", "feat_3"]},
    )
    percent_distinct, permtest_results = pertmutation_test_distinct_from_query_group(
        phe, "pert_iname=='DMSO'", "pert_iname", euclidean_metric
    )
    assert percent_distinct == pytest.approx(83.33333333333334)
    assert permtest_results.loc["DMSO"].item() == pytest.approx(0.9503)
    assert permtest_results.loc["Trt_1"].item() == pytest.approx(0.0001)
    assert permtest_results.sum().item() == pytest.approx(0.9510000000000001)


def test_distinctness_permutation_test_against_DMSO_extensive(
    synthetic_screening_dataset_2,
):
    euclidean_metric = non_ds_phenotypic_metrics["Euclidean"]
    phe = phenonaut.Phenonaut(
        synthetic_screening_dataset_2,
        metadata={"features": ["feat_1", "feat_2", "feat_3"]},
    )
    percent_distinct, permtest_results = pertmutation_test_distinct_from_query_group(
        phe,
        "pert_iname=='DMSO'",
        "pert_iname",
        euclidean_metric,
    )
    assert percent_distinct == pytest.approx(54.54545454545454)
    assert permtest_results.loc["DMSO"].item() == pytest.approx(0.8205)
    assert permtest_results.loc["Trt_6"].item() == pytest.approx(0.0838)
    assert permtest_results.sum().item() == pytest.approx(2.037)


def test_uniqueness_auroc(synthetic_screening_dataset_1):
    euclidean_metric = non_ds_phenotypic_metrics["Euclidean"]

    phe = phenonaut.Phenonaut(
        synthetic_screening_dataset_1,
        metadata={"features": ["feat_1", "feat_2", "feat_3"]},
    )
    auroc_results = auroc(phe, "pert_iname", phenotypic_metric=euclidean_metric)
    assert np.mean(
        [np.mean(vs) for vs in auroc_results["trt_aurocs"].values]
    ) == pytest.approx(0.9873697916666666)
