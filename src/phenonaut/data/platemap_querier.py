# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class PlatemapQuerier:
    plate_to_cpd_to_well_dict = {}
    platemap_files = None

    def __init__(
        self,
        platemap_directory: Union[str, Path],
        platemap_csv_files: Union[list, str, Path] = None,
        plate_name_before_underscore_in_filename=True,
    ):
        self.platemap_files = []
        if platemap_directory is None:
            if platemap_csv_files is None:
                raise FileNotFoundError("No platemaps given")
            if isinstance(platemap_csv_files, str):
                platemap_csv_files = [Path(platemap_csv_files)]
            if isinstance(platemap_csv_files, Path):
                platemap_csv_files = [platemap_csv_files]
            for i, f in enumerate(platemap_csv_files):
                if isinstance(f, str):
                    platemap_csv_files[i] = Path(f)
            for f in platemap_csv_files:
                if not platemap_csv_files[i].exists():
                    raise FileNotFoundError(f"Could not find specified platemap: {f}")
            self.platemap_files = platemap_csv_files
        else:
            if isinstance(platemap_directory, str):
                platemap_directory = Path(platemap_directory)
            if not platemap_directory.is_dir():
                raise FileNotFoundError(f"{platemap_directory} is not a directory")
            self.platemap_files = list(platemap_directory.glob("*.csv"))

        for csv_file in self.platemap_files:
            plate_name = csv_file.stem
            if plate_name_before_underscore_in_filename:
                plate_name = plate_name.split("_")[0]
            df = pd.read_csv(csv_file, index_col=0).rename(
                columns={"Unnamed: 0": "Column"}
            )
            self.plate_to_cpd_to_well_dict[plate_name] = self._platemap_to_dict(df)

        if len(self.plate_to_cpd_to_well_dict.keys()) == 0:
            raise FileNotFoundError("No suitable platemaps found")

    def _platemap_to_dict(self, plate):
        d = {}
        for column_label, column in plate.iteritems():
            for row, val in enumerate(column):
                if str(val) == "nan":
                    continue
                if val not in d.keys():
                    d[val] = []
                d[val].append(
                    (
                        plate.index.values[row],
                        plate.columns.values[int(column_label) - 1],
                    )
                )
        return d

    def get_compound_locations(self, cpd, plates: Union[str, list] = None):
        locations = []
        if isinstance(plates, str):
            plates = [plates]
        for platename, cpd_to_loc_dict in self.plate_to_cpd_to_well_dict.items():
            if plates is not None:
                if not platename in plates:
                    continue
            for location in cpd_to_loc_dict.get(cpd, []):
                locations.append((platename, location[0], location[1]))
        return locations
