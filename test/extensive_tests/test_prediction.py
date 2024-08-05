# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import shutil
import tempfile

from phenonaut.phenonaut import Phenonaut


def test_predict_iris_class_1_view(dataset_iris):
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
