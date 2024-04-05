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
from sklearn.datasets import load_iris

from ..data import Dataset
from .base import PackagedDataset


class MetadataJUMPMOACompounds(PackagedDataset):
    """DataFrame supplier for JUMP consortium MOA compound set

    This PackagedDataset provides access to a pd.DataFrame containing information
    on the JUMP MOA compound selection. Further information is available here:

    https://github.com/jump-cellpainting/JUMP-MOA

    Parameters
    ----------
    root : Union[Path, str]
        Local directory containing the prepared dataset. If the dataset is
        not found here and the argument download=True is not given, then an
        error is raised. If download=True and the processed dataset is
        absent, then it is downloaded the directory pointed at by the
        'raw_data' argument detailed below. If raw_data_dir is a
        non-absolute path, such as a single directory, then it is created as
        a subdirectory of this root directory.
    download : bool, optional
        If true and the processed dataset is not found in the root
        directory, then the dataset is downloaded and processed.
        By default False.
    raw_data_dir : Optional[Union[Path, str]], optional
        If downloading and preparing the dataset, then a directory for the
        raw data may be specified. If a non-absolute location is given, then
        it is created in a subdirectory of the root directory specified as
        the first argument. Absolute paths may be used to place raw
        datafiles and intermediates in another location, such as scratch
        disks etc, by default Path("raw_data").
    """

    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
    ):
        super().__init__(root, raw_data_dir)
        self.name = "JUMP MOA compounds"
        self.processed_h5_file = self.root / "JUMP_MOA_compounds_phenonaut.h5"

        # If the dataset is missing, get it
        self._call_if_file_missing(self.processed_h5_file, self._make, None)

        self.store = pd.HDFStore(self.processed_h5_file)

        self.register_df_key("MOA compounds")

        self.store.close()

    def _make(self, remove_intermediates=False):
        """Make the JUMP MOA DataFrame

        Parameters
        ----------
        remove_intermediates : bool, optional
            If true, then the downloaded archive is removed, by default False
        """
        self._download(
            "https://raw.githubusercontent.com/jump-cellpainting/JUMP-MOA/master/JUMP-MOA_compound_metadata.tsv",
            self.raw_data_dir / "JUMP-MOA_compound_metadata.tsv",
            mkdir=True,
            skip_if_exists=True,
        )

        from io import StringIO

        replace_df_string = "moa,new,old\nHMGCR inhibitor,delta-Tocotrienol,Compound1\nkinesin inhibitor,ispinesib,Compound2\nBCL inhibitor,ABT-737,Compound3\nPARP inhibitor,veliparib,Compound4\nIGF-1 inhibitor,NVP-AEW541,Compound5\ntricyclic antidepressant,dosulepin,Compound6\nFGFR inhibitor,BLU9931,Compound7\nphosphodiesterase inhibitor,quazinone,Compound8\n"
        replace_df = pd.read_csv(StringIO(replace_df_string), sep=",")

        df = (
            pd.read_csv(
                self.raw_data_dir / "JUMP-MOA_compound_metadata.tsv",
                sep="\t",
                dtype=str,
            )
            .set_index("pert_iname")
            .rename(index={old: new for old, new in replace_df[["old", "new"]].values})
        )
        h5_store = pd.HDFStore(self.processed_h5_file, "w", complevel=9)

        h5_store["/JUMP_MOA_compounds"] = df

        h5_store["/creation_date"] = pd.Series(str(datetime.datetime.now()))

        h5_store.close()

        self.register_df_key("JUMP_MOA_compounds")

    def __call__(self) -> pd.DataFrame:
        return self.get_df(None)

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
        if key == "" or key is None:
            df = store["/JUMP_MOA_compounds"].copy()
        else:
            df = store["/" + key].copy()
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
        return None


class MetadataBROADLincsCellPaintingMOAs(PackagedDataset):
    """DataFrame supplier for BROAD Lincs Cell Painting assigned MOAs

    This PackagedDataset provides access to a pd.DataFrame containing information
    on the BROAD institutes LINCS Cell Paiting compound MOA assignment.

    This data is located in the broadinstitute/lincs-cell-painting GitHub repository
    under metadata/moa/repurposing_simple.tsv.

    https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_simple.tsv

    Commentary on creation of this resource which may be useful is also available
    here:
    https://github.com/broadinstitute/lincs-cell-painting/issues/5

    Parameters
    ----------
    root : Union[Path, str]
        Local directory containing the prepared dataset. If the dataset is
        not found here and the argument download=True is not given, then an
        error is raised. If download=True and the processed dataset is
        absent, then it is downloaded the directory pointed at by the
        'raw_data' argument detailed below. If raw_data_dir is a
        non-absolute path, such as a single directory, then it is created as
        a subdirectory of this root directory.
    download : bool, optional
        If true and the processed dataset is not found in the root
        directory, then the dataset is downloaded and processed.
        By default False.
    raw_data_dir : Optional[Union[Path, str]], optional
        If downloading and preparing the dataset, then a directory for the
        raw data may be specified. If a non-absolute location is given, then
        it is created in a subdirectory of the root directory specified as
        the first argument. Absolute paths may be used to place raw
        datafiles and intermediates in another location, such as scratch
        disks etc, by default Path("raw_data").
    """

    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
    ):
        super().__init__(root, raw_data_dir)
        self.name = "JUMP MOA compounds"
        self.processed_h5_file = self.root / "JUMP_MOA_compounds_phenonaut.h5"

        # If the dataset is missing, get it
        self._call_if_file_missing(self.processed_h5_file, self._make, None)

        self.store = pd.HDFStore(self.processed_h5_file)

        self.register_df_key("BROAD_MOA_compounds")

        self.store.close()

    def _make(self, remove_intermediates=False):
        """Make the Broad MOA DataFrame

        Parameters
        ----------
        remove_intermediates : bool, optional
            If true, then the downloaded archive is removed, by default False
        """
        self._download(
            "https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_simple.tsv",
            self.raw_data_dir / "repurposing_simple.tsv",
            mkdir=True,
            skip_if_exists=True,
        )

        df = pd.read_csv(
            self.raw_data_dir / "repurposing_simple.tsv", sep="\t", dtype=str
        ).set_index("pert_iname")
        h5_store = pd.HDFStore(self.processed_h5_file, "w", complevel=9)

        h5_store["/Broad_MOA_compounds"] = df

        h5_store["/creation_date"] = pd.Series(str(datetime.datetime.now()))

        h5_store.close()

        self.register_df_key("Broad_MOA_compounds")

    def __call__(self) -> pd.DataFrame:
        return self.get_df(None)

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
        if key == "" or key is None:
            df = store["/Broad_MOA_compounds"].copy()
        else:
            df = store["/" + key].copy()
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
        return None
