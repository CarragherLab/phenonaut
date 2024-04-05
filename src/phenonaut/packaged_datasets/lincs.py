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

from ..data import Dataset
from .base import PackagedDataset


class LINCS_Cell_Painting(PackagedDataset):
    """LINCS Cell Painting Dataset - https://clue.io/

    This PackagedDataset provides supplies the following pd.DataFrames
    (queryable by calling the inherited ".keys" method):

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
        raw data may be specified. If a non-absolute location is given,
        then it is created in a subdirectory of the root directory specified
        as the first argument. Absolute paths may be used to place raw data
        files and intermediates in another location, such as scratch disks
        etc, by default Path("raw_data").
    """

    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
    ):
        """LINCS Cell Painting Dataset - https://clue.io/

        This PackagedDataset provides supplies the following pd.DataFrames
        (queryable by calling the inherited ".keys" method):

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
            raw data may be specified. If a non-absolute location is given,
            then it is created in a subdirectory of the root directory specified
            as the first argument. Absolute paths may be used to place raw data
            files and intermediates in another location, such as scratch disks
            etc, by default Path("raw_data").
        """

        super().__init__(root, raw_data_dir)
        self.name = "LINCS Cell Painting "
        self.processed_h5_file = self.root / "LINCS_CellPainting_phenonaut.h5"
        # If the dataset is missing, get it
        self._call_if_file_missing(self.processed_h5_file, self._make, None)

        store = pd.HDFStore(self.processed_h5_file)

        self.register_ds_key("ds")

        store.close()

    def _make(self, remove_intermediates=False):
        """Make main LINCS CellPainting HDF5 file

        Downloads the two compressed data files required (Batches 1 and 2),
        decompresses it, extracts the data into a pd.DataFrame, merges the
        sig_info data file which includes data on the perturbation type.

        Parameters
        ----------
        remove_intermediates : bool, optional
            If true, then the downloaded archive is removed, by default False
        """
        # XXX TODO change remove_intermediates to default to True.
        # XXX TODO change compression to 9.

        # Make sure the required files to make the dataset exist
        DownloadableFileInfo = namedtuple(
            "DownloadableFileInfo", ["filename", "compressed_filename", "url"]
        )
        required_files = [
            DownloadableFileInfo(
                "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt",
                "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
            ),
            DownloadableFileInfo(
                "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt",
                "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
            ),
            DownloadableFileInfo(
                "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt",
                "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
            ),
            DownloadableFileInfo(
                "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt",
                "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
            ),
            DownloadableFileInfo(
                "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt",
                "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
            ),
        ]
        for req_file in required_files:
            self._call_if_file_missing(
                self.root / req_file.filename,
                self._download,
                {
                    "source": req_file.url,
                    "destination": self.root / req_file.compressed_filename,
                    "extract": True,
                },
            )

        # Make sure the large GCTX file is present
        CMAP_GCTX_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
        CMAP_GCTX_GZ_FILENAME = (
            "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
        )
        self._call_if_file_missing(
            self.root / CMAP_GCTX_GZ_FILENAME,
            self._download,
            {
                "source": CMAP_GCTX_GZ_URL,
                "destination": self.root / (CMAP_GCTX_GZ_FILENAME + ".gz"),
                "extract": True,
            },
        )

        # Load in CMAP large dataset from GCTX file, reading it as an HDF5 file, and then walking the correct path.
        f = h5py.File(self.root / CMAP_GCTX_GZ_FILENAME)
        print(f"{datetime.datetime.now()} Extracting large matrix... ")
        self._df = pd.DataFrame(
            data=f["0"]["DATA"]["0"]["matrix"],
            columns=f["0"]["META"]["ROW"]["id"],
            index=f["0"]["META"]["COL"]["id"],
        )
        self._df.index = self._df.index.map(lambda x: x.decode("utf8"))
        self._df.columns = self._df.columns.map(lambda x: x.decode("utf8"))
        print(f"{datetime.datetime.now()} ..done")

        # Here, we generate cmap_phenonaut.h5, containing a merged df
        print(f"{datetime.datetime.now()} Merging")
        self._sig_info = pd.read_csv(
            self.root / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt",
            sep="\t",
            index_col=[0],
        )
        pert_dose = (
            self._sig_info["pert_idose"]
            .apply(lambda x: x.replace(" um", "").strip())
            .astype(float)
        )
        self._sig_info["pert_idose"] = np.where(pert_dose < 0, np.nan, pert_dose)
        self._df = self._df.merge(
            self._sig_info, left_index=True, right_index=True, copy=False
        )

        h5_store = pd.HDFStore(self.processed_h5_file, "w", complevel=0)
        h5_store["df"] = self._df

        h5_store["gene_info"] = pd.read_csv(
            self.root / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt",
            sep="\t",
            index_col=[0],
        )
        h5_store["sig_metrics"] = pd.read_csv(
            self.root / "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt",
            sep="\t",
            index_col=[0],
        )
        h5_store["pert_info"] = pd.read_csv(
            self.root / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt",
            sep="\t",
            index_col=[0],
        )

        h5_store["inst_info"] = pd.read_csv(
            self.root / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt",
            sep="\t",
            index_col=[0],
        )

        h5_store["creation_date"] = pd.Series(str(datetime.datetime.now()))
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
        if key == "landmark_gene_ids":
            store = pd.HDFStore(self.processed_h5_file)
            df = store["/gene_info"]
            return df.query("pr_is_lm == 1").index.to_numpy(dtype=str)

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
        if key == "ds":
            key = "df"
        store = pd.HDFStore(self.processed_h5_file)
        df = store["/" + key]
        features = [
            f
            for f in list(df.columns)
            if f
            not in [
                "pert_id",
                "pert_iname",
                "pert_type",
                "cell_id",
                "pert_idose",
                "pert_itime",
                "distil_id",
            ]
        ]
        ds = Dataset("cmap", df, {"features": features})
        store.close()
        return ds
