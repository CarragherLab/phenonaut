# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from phenonaut.output.heatmap import write_heatmap_from_df
from phenonaut.phenonaut import Phenonaut


def test_get_y_from_iris_dataset(dataset_iris):
    from phenonaut.predict.predict_utils import get_y_from_target

    assert get_y_from_target(dataset_iris).shape == (150,)


def test_write_heatmap():
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = pd.DataFrame(
            data=np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]),
            columns=["X1", "X2", "X3"],
            index=["Y1", "Y2", "Y3"],
        )
        write_heatmap_from_df(df, "Test heatmap", Path(tmp_dir) / "tmp_heatmap.png")
        assert (Path(tmp_dir) / "tmp_heatmap.png").stat().st_size > 10000


def test_instantiate_dave():
    from phenonaut.predict.default_predictors.pytorch_models.dave import _DAVE_model

    d = _DAVE_model(view_sizes=(100, 150), embedding_size=32, n_hidden=2)
    assert d.topology is not None


def test_predict_iris_class_1_view(phenonaut_object_iris_2_views, dataset_iris):
    dirpath = tempfile.mkdtemp()
    print(dirpath)
    phe = Phenonaut(dataset_iris)
    import phenonaut.predict

    phenonaut.predict.profile(phe, dirpath)
    shutil.rmtree(dirpath)


def test_predict_iris_class_2_views(phenonaut_object_iris_2_views):
    dirpath = tempfile.mkdtemp()
    print(dirpath)
    phe = phenonaut_object_iris_2_views
    import phenonaut.predict

    phenonaut.predict.profile(phe, dirpath)
    shutil.rmtree(dirpath)
