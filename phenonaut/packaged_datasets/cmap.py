# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from collections import namedtuple
from pathlib import Path
import pathlib
from typing import List, NamedTuple, Optional, Union
from ..data import Dataset
import pandas as pd
from .base import PackagedDataset
import gzip, tarfile
import shutil
import pandas as pd
import numpy as np
import h5py
import datetime


class CMAP(PackagedDataset):
    """CMAP - ConnectivityMap dataset - https://clue.io/

    CMAP is a repository of L1000 profiles measured from small molecule and
    crisper perturbations. This CMAP packaged dataset supplies an interface
    to querying this data as a pd.DataFrame.

    Data supplied by CMAP is in their own GCTX format files which are HDF5
    files with data residing in specific paths. Rather than have Phenonaut
    rely on their supplied library, we simply read the file with standard
    HDF5 tools. Additionally, we merge sig_info data, assigning
    perturbation information, bringing in the following columns:
    'pert_id', 'pert_iname', 'pert_type', 'cell_id', 'pert_idose',
    'pert_itime', and 'distil_id'.
    
    in the pert_type column, and taking the following values:
    
    ctl_vehicle : DMSO

    trt_cp      : compound treatment

    trt_xpr     : crisper treatment

    If the pert_type is trt_cp, then pert_idose gives the compound
    concentration.  In the supplied cmap data, the field is a string
    containing for example, "3.33 um". This dataloader changes this to a
    float field without the µM (um as written) prefix - units are µM. If
    the pert_type is ctl_vehicle or trt_xpr, then CMAP supplied data with
    -666 in the pert_idose field. This is changed to np.nan.

    Further field information can be found here:
    https://clue.io/connectopedia/glossary
    
    This PackagedDataset provides supplies the following pd.DataFrames
    (queryable by calling the inherited ".keys" method):
    
    * creation_date
        the date on which the h5 file was written.
    * df
        the main dataframe containing L1000 data, merged with sig_info to give a more complete view of the dataset.
    * gene_info
        Information on the genes - allows translation of L1000 gene number in df to gene name and more.
    * pert_info
        Perturbation info.  Using the df, which contains the merged sig_info data, compound ids can be used to query this dataframe and molecule smiles etc returned.
    * sig_metrics
        Additional metrics on the profiles recorded in df.
    * inst_info
        Information on plate barcodes etc from which profiles derived.

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
        as the first argument. Absolute paths may be used to place raw
        datafiles and intermediates in another location, such as scratch
        disks etc, by default Path("raw_data").
    landmark_only : bool
        If True, then only return landmark genes, essentially removing all
        inferred gene abundances. This is likely the most useful for the
        majority of tasks ashighly correlated abundances simply adds to
        colinearity.
    """
    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
        landmark_only:bool = True,
    ):

        super().__init__(root, raw_data_dir)
        self.name="CMAP"
        self.landmark_only=landmark_only
        self.processed_h5_file=self.root / "cmap_phenonaut.h5"
        # If the dataset is missing, get it
        self._call_if_file_missing(self.processed_h5_file, self._make_merged_data_and_sig_info, None)

        store=pd.HDFStore(self.processed_h5_file)
        required_keys=['/creation_date', '/df', '/gene_info', '/pert_info', '/sig_metrics', '/inst_info']
        if not all(rk in store.keys() for rk in required_keys):
            raise KeyError(f"{self.processed_h5_file.name} does not contain the correct df keys, needed {required_keys}, but got {store.keys()}")
        
        for k in [key for key in required_keys if not key=="/df"]:
            self.register_df_key(k[1:])
        self.register_df_key("landmark_gene_ids")

        self.register_ds_key('ds')

        store.close()
        
    def _make_merged_data_and_sig_info(self, remove_intermediates=False):
        """Make main CMAP HDF5 file

        Downloads the large compressed data file, decompresses it, extracts the
        data into a pd.DataFrame, merges the sig_info data file which includes
        data on the perturbation type.

        Parameters
        ----------
        remove_intermediates : bool, optional
            If true, then the downloaded archive is removed, by default False
        """
        # XXX TODO change remove_intermediates to default to True.
        # XXX TODO change compression to 9.

        # Make sure the required files to make the dataset exist
        DownloadableFileInfo = namedtuple("DownloadableFileInfo", ["filename", "compressed_filename", "url"])
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
        CMAP_GCTX_GZ_FILENAME = "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
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
        pert_dose = self._sig_info["pert_idose"].apply(lambda x: x.replace(" um", "").strip()).astype(float)
        self._sig_info["pert_idose"] = np.where(pert_dose < 0, np.nan, pert_dose)
        self._df = self._df.merge(self._sig_info, left_index=True, right_index=True, copy=False)

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

    def get_df(self, key:str)->pd.DataFrame:
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
        if key=="landmark_gene_ids":
            store=pd.HDFStore(self.processed_h5_file)
            df = store["/gene_info"]
            return df.query("pr_is_lm == 1").index.to_numpy(dtype=str)

        store=pd.HDFStore(self.processed_h5_file)
        df=store["/"+key]
        store.close()
        return df

    def get_ds(self, key:str)->Dataset:
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
        if key=="ds":
            key="df"
        store=pd.HDFStore(self.processed_h5_file)
        pert_info_columns=["pert_id","pert_iname","pert_type","cell_id","pert_idose","pert_itime","distil_id"]
        if self.landmark_only:
            columns_needed=list(pert_info_columns)
            landmark_gene_list=self.get_df("landmark_gene_ids")
            columns_needed.extend(landmark_gene_list)
            df=store["/"+key][columns_needed]
        else:
            df=store["/"+key]
        features=[f for f in list(df.columns) if f not in pert_info_columns]
        ds=Dataset("cmap", df, {'features':features})
        store.close()
        return ds