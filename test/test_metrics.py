import phenonaut
import numpy as np
from phenonaut.phenonaut import Phenonaut
from phenonaut.metrics import non_ds_phenotypic_metrics

def test_phenotypic_metrics(small_2_plate_df):
    """Test a selection of phenotypic metrics

    """
    print(non_ds_phenotypic_metrics)
    assert False
