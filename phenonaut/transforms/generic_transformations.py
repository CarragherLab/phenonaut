# Copyright © The University of Edinburgh, 2023.
# Development has been supported by GSK.

from phenonaut.data import Dataset
from phenonaut.transforms.transformer import Transformer
import numpy as np

# Inserted as an example of making a custom transformer from a simple function
log2_transformer = Transformer(np.log2)

# Alternatively, the class
class Log2(Transformer):
    def __init__(self):
        super().__init__(np.log2, new_feature_names="log2_")
