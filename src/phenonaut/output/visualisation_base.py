# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import json
from pathlib import Path
from typing import Optional, Union

import yaml

from phenonaut.utils import load_dict


class PhenonautVisualisation:
    """Base class for Phenonaut visualisations.

    Constructor allows supply of a plot_config dictionary, which is then stored
    in object under the name 'config'.  Out of this dictionary, if it exists,
    then a nested 'plot_markers' dictionary is extracted and placed into the
    class' markers member variable.


    Parameters
    ----------
    config : Optional[Union[Path, str, dict]], optional
        Configuration dictionary, which will be stored in config member
        variable. If None, then an empty dictionary is initialised as the
        config member variable. By default, None
    """

    def __init__(self, plot_config: Optional[Union[Path, str, dict]] = None):
        markers = {}
        self.config = load_dict(plot_config, cast_none_to_dict=True)
        for k, v in self.config.get("plot_markers", {}).items():
            self.markers[k] = v
