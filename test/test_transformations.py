# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.
import pandas as pd
import numpy as np
import phenonaut
import phenonaut.transforms


def test_transforms_standardscaler(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    stdscaler = phenonaut.transforms.StandardScaler()
    stdscaler(phe.ds)
    assert (
        phe.df["StdScaler_feat_3"]
        - pd.Series([-0.505442, -0.424355, 0.994667, 1.724450, -0.910877, -0.878443])
    ).abs().mean() < 1e-6


def test_transforms_pca(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    t_pca = phenonaut.transforms.PCA()
    t_pca(phe.ds)
    assert (
        phe.df["PC_2"]
        - pd.Series([0.033607, -0.005505, 0.342870, -0.276157, -0.037338, -0.057477])
    ).abs().mean() < 1e-6


def test_transforms_zca(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    t_zca = phenonaut.transforms.ZCA()
    t_zca(phe.ds)
    assert (
        phe.df["ZCA_feat_3"]
        - pd.Series([-0.386266, -0.243655, -0.784378, 1.998697, -0.365108, -0.219291])
    ).abs().mean() < 1e-6


def test_transforms_tnse(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    constructor_kwargs = {}
    constructor_kwargs["perplexity"] = 2
    t_tsne = phenonaut.transforms.TSNE(constructor_kwargs, ndims=2)
    t_tsne(phe.ds)
    # Due to the stochastic nature of t-SNE, would not be a good idea to try to compare expected values
    assert len(phe.ds.features) == 2


def test_transforms_umap(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    t_umap = phenonaut.transforms.UMAP()
    t_umap(phe.ds)
    assert len(phe.ds.features) == 2


def test_transforms_custom_callable_square(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    from numpy import all as np_all
    from numpy import square

    f1_series = phe.ds.data.values[:, 0]
    t_square = phenonaut.transforms.Transformer(square)
    t_square(phe.ds)
    assert np_all(f1_series * f1_series == phe.ds.data.values[:, 0])


def test_transforms_robustmad(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    robust_mad = phenonaut.transforms.RobustMAD()

    # Enumerate all ways of calling robustmad:
    robust_mad(phe.ds)
    robust_mad.fit_transform(phe.ds)
    robust_mad.transform(phe.ds)
    robust_mad.fit(phe.ds)
    robust_mad(phe.ds, groupby="BARCODE")
    assert (
        phe.df["RobustMAD_RobustMAD_RobustMAD_RobustMAD_feat_3"]
        - pd.Series([-0.063631, 0.063631, 2.290720, 3.436080, -0.699942, -0.649037])
    ).abs().sum() < 1e-6


def test_transforms_log2(small_2_plate_df):
    phe = phenonaut.Phenonaut(small_2_plate_df)
    log2_t = phenonaut.transforms.Log2()
    log2_t(phe.ds)
    assert (
        phe.df["log2_feat_3"]
        - pd.Series([0.378512, 0.584963, 2.321928, 2.765535, -1.736966, -1.395929])
    ).abs().sum() < 2e-6

