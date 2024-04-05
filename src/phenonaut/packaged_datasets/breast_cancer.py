# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import datetime
import gzip
import pathlib
import shutil
import tarfile
from collections import namedtuple
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

import h5py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from ..data import Dataset
from .base import PackagedDataset


class BreastCancer(PackagedDataset):
    """Breast Cancer Dataset from scikit-learn

    This PackagedDataset provides the Breast Cancer dataset from
    scikit-learn. This is also known as the Breast cancer Wisconsin
    (diagnostic) dataset.
    See the scikit-learn user guide for more information:
    https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
    Original dataset information available at:
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

    Contains 569 unique samples breast cancer fine needle aspirates, each
    with 30 features, and one target of 0 or 1, denoting benign or
    malignant respectively.

    Parameters
    ----------
    root : Union[Path, str]
        Local directory containing the prepared dataset. If the dataset is
        not found here and the argument download=True is not given, then an
        error is raised. If download=True and the processed dataset is
        absent, then it is downloaded the directory pointed at by the
        'raw_data' argument detailed below. If raw_data_dir is a
        non-absolute path, such as a single directory,
        then it is created as a subdirectory of this root directory.
    download : bool, optional
        If true and the processed dataset is not found in the root
        directory, then the dataset is downloaded and processed.
        By default False.
    raw_data_dir : Optional[Union[Path, str]], optional
        If downloading and preparing the dataset, then a directory for the
        raw data may be specified. If a non-absolute location is given,
        then it is created in a subdirectory of the root directory
        specified as the first argument. Absolute paths may be used to
        place raw datafiles and intermediates in another location, such as
        scratch disks etc, by default Path("raw_data").
    """

    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
    ):
        """Breast Cancer Dataset from scikit-learn

        This PackagedDataset provides the Breast Cancer dataset from
        scikit-learn. This is also known as the Breast cancer Wisconsin
        (diagnostic) dataset.
        See the scikit-learn user guide for more information:
        https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
        Original dataset information available at:
        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        Contains 569 unique samples breast cancer fine needle aspirates, each
        with 30 features, and one target of 0 or 1, denoting benign or
        malignant respectively.

        Parameters
        ----------
        root : Union[Path, str]
            Local directory containing the prepared dataset. If the dataset is
            not found here and the argument download=True is not given, then an
            error is raised. If download=True and the processed dataset is
            absent, then it is downloaded the directory pointed at by the
            'raw_data' argument detailed below. If raw_data_dir is a
            non-absolute path, such as a single directory,
            then it is created as a subdirectory of this root directory.
        download : bool, optional
            If true and the processed dataset is not found in the root
            directory, then the dataset is downloaded and processed.
            By default False.
        raw_data_dir : Optional[Union[Path, str]], optional
            If downloading and preparing the dataset, then a directory for the
            raw data may be specified. If a non-absolute location is given,
            then it is created in a subdirectory of the root directory
            specified as the first argument. Absolute paths may be used to
            place raw datafiles and intermediates in another location, such as
            scratch disks etc, by default Path("raw_data").
        """

        super().__init__(root, raw_data_dir)
        self.name = "Breast cancer dataset"
        self.processed_h5_file = self.root / "breast_cancer_Wisconsin_phenonaut.h5"

        # If the dataset is missing, get it
        self._call_if_file_missing(self.processed_h5_file, self._make, None)

        self.store = pd.HDFStore(self.processed_h5_file)

        self.register_ds_key("Breast_cancer_dataset")

        self.store.close()

    def _make(self, remove_intermediates=False):
        """Make IRIS HDF5 file

        Parameters
        ----------
        remove_intermediates : bool, optional
            If true, then the downloaded archive is removed, by default False.
        """
        df = load_breast_cancer(as_frame=True).frame
        h5_store = pd.HDFStore(self.processed_h5_file, "w", complevel=9)

        h5_store["/Breast_cancer_dataset"] = df

        h5_store["/creation_date"] = pd.Series(str(datetime.datetime.now()))

        h5_store.close()

    def get_df(self, key: str) -> pd.DataFrame:
        """Get supporting dataframe

        Parameters
        ----------
        key : str
            Key of pd.DataFrame.

        Returns
        -------
        pd.DataFrame
            Requested pd.DataFrame from h5 store
        """
        store = pd.HDFStore(self.processed_h5_file)
        df = store["/" + key]
        store.close()
        return df

    def get_ds(self, key: str) -> Dataset:
        """Get supporting dataframe

        Parameters
        ----------
        key : str
            Key of Phenonaut Dataset.

        Returns
        -------
        Dataset
            Requested Phenonaut Dataset from h5 store, with the correctly set
            features and metadata
        """
        store = pd.HDFStore(self.processed_h5_file)
        df = store["/" + key]
        features = [f for f in list(df.columns) if f not in ["target"]]
        ds = Dataset("Breast_Cancer", df, {"features": features})
        store.close()
        return ds
