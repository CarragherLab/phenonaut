# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from time import sleep
import pytest
import phenonaut
from sklearn.datasets import load_iris
from iris import *
from phenonaut.data.dataset import Dataset
import pandas as pd
import numpy as np
from phenonaut.phenonaut import Phenonaut
from io import StringIO

def test_get_features_and_non_features(dataset_iris):
    features = dataset_iris.features
    print(features)
    non_features = dataset_iris.get_non_feature_columns()
    assert sorted(features + non_features) == sorted(dataset_iris.df.columns.values)

def test_new_aggregated_dataset():
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
    
    df=pd.DataFrame({
        'ROW':[1,1,1,1,1,1],
        'COLUMN':[1,1,1,1,2,2],
        'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
        'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
        'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
        'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
        'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
        'FOV':[1,2,1,2,1,2]})
    
    phe=phenonaut.Phenonaut(df, metadata={'features_prefix':'feat_'})
    new_ds=phe.ds.new_aggregated_dataset(["ROW", "COLUMN", "BARCODE"])
    print(new_ds.df)
    assert set(new_ds.features)==set(['feat_1','feat_2','feat_3'])
    assert abs(new_ds.df['feat_1'][0]-1.25<1e-6)
    assert np.abs(new_ds.df['feat_1'][1]-5.70<1e-6)
    assert np.abs(new_ds.df['feat_1'][2]-0.15<1e-6)
    assert np.abs(new_ds.df['feat_2'][0]-1.3<1e-6)
    assert np.abs(new_ds.df['feat_2'][1]-5.6<1e-6)
    assert np.abs(new_ds.df['feat_2'][2]-0.2<1e-6)
    assert np.abs(new_ds.df['feat_3'][0]-1.4<1e-6)
    assert np.abs(new_ds.df['feat_3'][1]-5.9<1e-6)
    assert np.abs(new_ds.df['feat_3'][2]-0.34<1e-6)
    

def test_iris_packageddataset():
    import tempfile

    tmpdir=tempfile.gettempdir()
    from phenonaut.packaged_datasets import Iris, Iris_2_views
    phe=Phenonaut(Iris(tmpdir))
    assert len(phe.keys())==1


def test_feature_selection_when_loading_csv():
    import tempfile
    with tempfile.NamedTemporaryFile(mode = "w") as tmp:
        tmp.write("row, column, feature1, feature2, derived_feature3, feature10, feature003\n1,1,1,2,3,4,5\n2,1,6,7,8,9,10")
        tmp.flush()
        phe=Phenonaut()
        phe.load_dataset(tmp.name,tmp.name, {'features_prefix':"feat"})
        assert phe.ds.features==['feature1', 'feature2', 'feature003', 'feature10']
        
    with tempfile.NamedTemporaryFile(mode = "w") as tmp:
        tmp.write("sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target\n5.4,3.4,1.7,0.2,0\n7.2,3.0,5.8,1.6,2\n6.4,2.8,5.6,2.1,2\n4.8,3.1,1.60.2,0\n5.6,2.5,3.9,1.1,1")
        tmp.flush()
        phe=Phenonaut()
        phe.load_dataset(tmp.name,tmp.name, {'features_regex':".*(width|length).*"})
        assert phe.ds.features==['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)']


def test_replace_string_in_column():
    """Here we test a df as follows:
    ROW	COLUMN	BARCODE	feat_1	feat_2	feat_3	filename	FOV
    1	1	    Plate1	1.2	    1.2	    1.3	    fileA.png	1
    1	1	    Plate1	1.3	    1.4	    1.5	    FileB.png	2
    1	1	    Plate2	5.2	    5.1	    5	    FileC.png	1
    1	1	    Plate2	6.2	    6.1	    6.8	    FileD.png	2
    1	2	    Plate1	0.1	    0.2	    0.3	    fileE.png	1
    1	2	    Plate1	0.2	    0.2	    0.38    FileF.png	2
    
    """

    df=pd.read_csv(StringIO(
"""ROW,COLUMN,BARCODE,feat_1,feat_2,feat_3,filename,FOV
1,1,Plate1,1.2,1.21.3,fileA.png,1
1,1,Plate1,1.3,1.41.5,FileB.png,2
1,1,Plate2,5.2,5.15,FileC.png,1
1,1,Plate2,6.2,6.16.8,FileD.png,2
1,2,Plate1,0.1,0.20.3,fileE.png,1
1,2,Plate1,0.2,0.20.38,FileF.png,2
"""))
    
    phe=phenonaut.Phenonaut(df, metadata={'features_prefix':'feat_'})
    phe[-1].replace_str("BARCODE", "Plate", "P")
    assert all(elem in ['P1', 'P2'] for elem in phe.df.BARCODE.unique())

