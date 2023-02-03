# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.
import pytest
import phenonaut
import pandas as pd
import phenonaut.transforms
from pathlib import Path
def test_transforms_standardscaler():
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=phenonaut.Phenonaut(df)
    stdscaler=phenonaut.transforms.StandardScaler()
    stdscaler(phe.ds)
    assert abs(phe.df.iloc[3,4]-1.589652)<1e-5

def test_transforms_pca():
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=phenonaut.Phenonaut(df)
    t_pca=phenonaut.transforms.PCA()
    t_pca(phe.ds)
    assert abs(phe.ds.data.iloc[3,0]-6.829846)<1e-5

def test_transforms_zca():
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})

    phe=phenonaut.Phenonaut(df)
    t_zca=phenonaut.transforms.ZCA()
    t_zca(phe.ds)
    print(phe.ds.data)
    assert abs(phe.ds.data.iloc[3,0]-0.299764)<1e-5

def test_transforms_tnse():
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=phenonaut.Phenonaut(df)
    t_tsne=phenonaut.transforms.TSNE()
    t_tsne(phe.ds)
    # Due to the stochastic nature of t-SNE, would not be a good idea to try to compare expected values
    assert len(phe.ds.features)==2

    
def test_transforms_umap():
    from phenonaut import Phenonaut
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=Phenonaut(df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    t_umap=phenonaut.transforms.UMAP()
    t_umap(phe.ds)
    assert len(phe.ds.features)==2


def test_transforms_custom_callable_square():
    from numpy import square, all as np_all
    from phenonaut import Phenonaut
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=Phenonaut(df, "PheSmallDS", metadata={'features_prefix':'feat_'})
    f1_series=phe.ds.data.values[:,0]
    t_square=phenonaut.transforms.Transformer(square)
    t_square(phe.ds)
    assert np_all(f1_series*f1_series==phe.ds.data.values[:,0])

def test_transforms_robustmad():
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=phenonaut.Phenonaut(df)
    robust_mad=phenonaut.transforms.RobustMAD()
    robust_mad(phe.ds)
    assert abs(phe.df.iloc[3,4]-4.363636)<1e-5