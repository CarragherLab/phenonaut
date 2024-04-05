# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import datetime
import shutil
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Type, Union

import h5py
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv

from ..data import Dataset
from .base import PackagedDataset


class TCGA(PackagedDataset):
    """TCGA - The Cancer Genome Atlas, packaged dataset

    The TCGA dataset captures a snapshot of The Cancer Genome Atlas, from
    the TCGA website:

    https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga

        "The Cancer Genome Atlas (TCGA), a landmark cancer genomics program,
        molecularly characterized over 20,000 primary cancer and matched
        normal samples spanning 33 cancer types. This joint effort between
        NCI and the National Human Genome Research Institute began in 2006,
        bringing together researchers from diverse disciplines and multiple
        institutions."

    Processing of the dataset occurs in a manner the same as that described
    by Lee in:

        Lee, Changhee, and Mihaela Schaar. "A Variational Information
        Bottleneck Approach to Multi-Omics Data Integration." International
        Conference on Artificial Intelligence and Statistics. PMLR, 2021.

    Although we have taken the decision to process the dataset into 10 PCA
    dimensions using linear PCA, although the number of PCA dimensions can
    be changed using the num_pca_dims argument to this constructor.

    Additionally no PCA can be applied and custom transformation in the form of a callable used by passing it to custom_transformation_func_and_name_tuple.

    Datasets representing clinical_decision, RPPA, miRNA, methylation,
    and mRNA are generated. These datasets are processed and saved in the
    HDF5 file format, writing files of the format
    "tcga_pca{num_pca_dims}_phenonaut.h5" and
    "tcga_pca{num_pca_dims}_metadata_phenonaut.h5" where {num_pca_dims} is
    the requested number of PCA dimensions.

    Steps undertaken in dataset preparation are as follows:

    #. Download the dataset. This downloads 186 .tar.gz files totalling ~20 GB. These are then extracted, taking ~32 GB.
    #. Files representing different tumour types for each view are merged, and empty columns removed. These files are then saved as intermediates.

    On modest 2020 hardware, processing:

    * clinical_decisions takes ~2 secs, generating 2.5 MB intermediate
    * RPPA takes ~4 secs, generating 29 MB intermediate
    * miRNA takes ~ 5 secs, generating 102 B intermediate
    * methylation takes ~12.5 mins, generating 3.6GB intermediate
    * mRNA takes 10 mins, generating 4 MB intermediate files

    Parameters
    ----------
    root : Union[Path, str]
        Local directory containing the prepared dataset. If the dataset is
        not found here and the argument download=True is not given, then an
        error is raised. If download=True and the processed dataset is
        absent, then it is downloaded the directory pointed at by the
        'raw_data' argument detailed below. If raw_data_dir is a
        non-absolute path, such as a single directory, then it is createdas
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
    rm_downloaded_data : bool
        If creating the dataset, and this is True, then downloaded data raw
        TCGA data (archives) will be deleted, by default True.
    rm_intermediates : bool
        If creating the dataset, and this is True, then intermediate data
        generated from the extraction of TCGA archives will be deleted,
        by default True.
    prediction_target: str, optional
        Often we are want to make predictions on datasets using labels or
        targets captured by TCGA and placed in the clinical_decisions
        dataframe, an example is the commonly used days_to_death column.
        This argument may be any column within that DataFrame (queryable by
        calling get_clinical_decisions_columns)
        such as days_to_death, tumor_tissue_site, gender etc additionally
        to these column names, the string 'years_to_death' can also be used,
        which will operate on days_to_death divided by 365.25,
        by default None.
    num_pca_dims: int, optional
        The number of linear principal components to use in dimensionality
        reduction. By default 10.
    vif_filter_cutoff : float, optional
        Apply a VIF (variance inflation factor) cutoff, removing features
        with a VIF score greater than this value. This has the effect or
        removing features which have a high degree of colinearity. If
        vif_filter_cutoff is None, then no vif filter is applied. A good
        default choice for this value is 5.0. If a
        custom_transformation_func_and_name_tuple as defined below is given,
        then the vif filter is ignored, to include it, you may combine ZIF
        filtering in the custom function.
    custom_transformation_func_and_name_tuple: Optional[tuple[Callable, str]]
        If PCA is not the preferred transformation to be applied to the data,
        then the user may provide their own in a tuple. The first tuple
        element should be the callable function, and the second a unique
        name/identifier which will be used to uniquely identify the saved
        dataset. Whereas datasets using the default PCA dimensionality
        reduction technique are named:
        "tcga_pca{num_pca_dims}_phenonaut.h5
        Datasets named by custom callables will be named:
        "tcga_{custom_callable_id}_phenonaut.h5
        where custom_callable_id is the second element of the
        custom_transformation_func_and_name_tuple tuple. If None, then no
        customtransformation or dimensionality reduction is performed,
        instead using the standard scalar, followed by PCA approach as
        described above. If a custom_transformation_func_and_name_tuple,
        then vif_filter_cutoff has no effect, and it is as if it is set to
        None. By default None.
    """

    # NamedTuple which helps processing of downloaded files.
    TCGA_MetadataTuple = namedtuple(
        "TCGA_MetadataTuple",
        [
            "files",  # Files containing data to be merged
            "load_csv_kwargs",  # Custom Pandas kwargs for some datasets as a row needs to be skipped.
            "output_file_name",  # Merged dataset target filename
            "header_offset",  # Header row may not be 0 in some cases.
            "treatment_id",  # All are something like "Hybridization REF", but small changes are present in raw TCGA data
            "treatment_lambda",  # TreatmentID transformation, usually convert to lowercase, sometimes remove characters.
        ],
    )

    def __init__(
        self,
        root: Union[Path, str],
        download: bool = False,
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        rm_downloaded_data: bool = True,
        rm_intermediates: bool = True,
        prediction_target: Optional[str] = None,
        num_pca_dims: Union[int, None] = 10,
        vif_filter_cutoff: Optional[float] = None,
        custom_transformation_func_and_name_tuple: Optional[
            tuple[Callable, str]
        ] = None,
    ):
        # List of clinical tumor types within TCGA
        self.tumor_list = [
            "ACC",
            "BLCA",
            "BRCA",
            "CESC",
            "CHOL",
            "COAD",
            "COADREAD",
            "DLBC",
            "ESCA",
            "FPPP",
            "GBM",
            "GBMLGG",
            "HNSC",
            "KICH",
            "KIPAN",
            "KIRC",
            "KIRP",
            "LAML",
            "LGG",
            "LIHC",
            "LUAD",
            "LUSC",
            "MESO",
            "OV",
            "PAAD",
            "PCPG",
            "PRAD",
            "READ",
            "SARC",
            "SKCM",
            "STAD",
            "STES",
            "TGCT",
            "THCA",
            "THYM",
            "UCEC",
            "UCS",
            "UVM",
        ]
        super().__init__(root, raw_data_dir)
        self.prediction_target = prediction_target
        self.name = "TCGA"
        self.vif_filter_cutoff = vif_filter_cutoff
        # Handle the supply of custom transformation function

        if custom_transformation_func_and_name_tuple is not None and not isinstance(
            custom_transformation_func_and_name_tuple, tuple
        ):
            raise ValueError(
                "custom_transformation_func_and_name_tuple must be a tuple of the form tuple[Callable, str] where str is a short descriptive string to be used in the filename - such as 'UMAP' or 'LDA'"
            )

        if custom_transformation_func_and_name_tuple is None:
            self.custom_transformation_func = None
            self.processed_h5_file = (
                self.root
                / f"tcga_pca{num_pca_dims}_VIF{self.vif_filter_cutoff}_phenonaut.h5"
            )
            self.metadata_h5_file = (
                self.root
                / f"tcga_pca{num_pca_dims}_VIF{self.vif_filter_cutoff}_metadata_phenonaut.h5"
            )
            self.num_pca_dims = num_pca_dims
        else:
            self.custom_transformation_func = custom_transformation_func_and_name_tuple[
                0
            ]
            self.processed_h5_file = (
                self.root
                / f"tcga_{custom_transformation_func_and_name_tuple[1]}_phenonaut.h5"
            )
            self.metadata_h5_file = (
                self.root
                / f"tcga_{custom_transformation_func_and_name_tuple[1]}_metadata_phenonaut.h5"
            )

        # If the dataset is missing, get it
        self._call_if_file_missing(
            self.processed_h5_file,
            self._make_dataset,
            {
                "download": download,
                "rm_downloaded_data": rm_downloaded_data,
                "rm_intermediates": rm_intermediates,
            },
        )

        if not self.metadata_h5_file.exists():
            raise FileNotFoundError(
                f"{self.metadata_h5_file} is not present. Something has gone wrong. Suggest deleting {self.metadata_h5_file.parent} directory and starting again"
            )

        store = pd.HDFStore(self.processed_h5_file)
        for key in store.keys():
            if key != "/clinical_decisions":
                self.register_ds_key(key[1:])
        self.register_df_key("clinical_decisions")
        self.clinical_decisions_df = store["clinical_decisions"].set_index(
            "Hybridization REF"
        )
        store.close()

    def _make_dataset(
        self,
        download: bool = False,
        rm_downloaded_data: bool = True,
        rm_intermediates: bool = True,
    ):
        if not download:
            print(
                "Attempting to preprocess the dataset, however, download = False. Dataset creation will fail if the files are not already present."
            )
        self.tmp_dir = self.raw_data_dir / Path("tmp")
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True)
        if download:
            self.__download_TCGA()
        self.__decompress_TCGA()
        self.__merge_data()
        self.__perform_pca_write_h5()

    def __download_TCGA(self):
        remote_base_url = (
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/"
        )
        filename_suffixes = [
            "Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz",
            "miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
            "mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
            "RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz",
            "Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz",
        ]

        download_tasks = []
        print(f"{download_tasks=}")
        for tumor in self.tumor_list:
            filename_prefix = f"gdac.broadinstitute.org_{tumor}"
            for filename in [f"{filename_prefix}.{fns}" for fns in filename_suffixes]:
                if not (self.raw_data_dir / Path(filename)).exists():
                    print(
                        f"Download task: {remote_base_url}{tumor}/20160128/{filename}"
                    )
                    download_tasks.append(
                        (
                            f"{remote_base_url}{tumor}/20160128/{filename}",
                            self.raw_data_dir / filename,
                        )
                    )
        print(f"Downloading {len(download_tasks)} files")
        # According to the format and patterns of deposited TCGA data, the 4 files
        # below should exist, but they are not present on the server.  Including
        # them as non-critical allows checking if they are available without raising
        # an exception when they are not downloaded.
        non_critical_urls = [
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/FPPP/20160128/gdac.broadinstitute.org_FPPP.Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz",
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/FPPP/20160128/gdac.broadinstitute.org_FPPP.mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/FPPP/20160128/gdac.broadinstitute.org_FPPP.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz",
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/LAML/20160128/gdac.broadinstitute.org_LAML.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz",
        ]
        self._batch_download(download_tasks, non_critical_urls=non_critical_urls)

    def __decompress_TCGA(self):
        # Extract compressed files
        for file in [
            f
            for f in self.raw_data_dir.glob("*.gz")
            if not Path(str(f)[: -len(".tar.gz")]).exists()
        ]:
            # Some files like gdac.broadinstitute.org_LIHC.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz
            # do not unpack themselves to the expected
            # Expected: gdac.broadinstitute.org_LIHC.RPPA_AnnotateWithGene.Level_3.2016012800.0.0
            # Actual  : gdac.broadinstitute.org_LIHC.RPPA_AnnotateWithGene.Level_3.2016071400.0.0
            # So a single attempt to extract all is good enough.
            print(f"Extracing {file}")
            shutil.unpack_archive(file, file.parent)

    def __merge_data(self):
        tcga_raw_data_metadata = {
            "clinical_decision": self.TCGA_MetadataTuple(
                list(self.raw_data_dir.glob("**/*.clin.merged.picked.txt")),
                {},
                "clinical_decisions.csv",
                1,
                "Hybridization REF",
                lambda x: x.lower(),
            ),
            "RPPA": self.TCGA_MetadataTuple(
                list(self.raw_data_dir.glob("**/*.rppa.txt")),
                {},
                "RPPA.csv",
                1,
                "Composite.Element.REF",
                lambda x: x.lower()[:12],
            ),
            "miRNA": self.TCGA_MetadataTuple(
                list(self.raw_data_dir.glob("**/*.miRseq_RPKM_log2.txt")),
                {},
                "miRNAseq_RPKM_log2.csv",
                1,
                "gene",
                lambda x: x.lower()[:-3],
            ),
            "methylation": self.TCGA_MetadataTuple(
                list(self.raw_data_dir.glob("**/*.meth.by_mean.data.txt")),
                {"skiprows": [1]},
                "methylation.csv",
                2,
                "Hybridization REF",
                lambda x: x.lower()[:-3],
            ),
            "mRNA": self.TCGA_MetadataTuple(
                list(
                    self.raw_data_dir.glob(
                        "**/*.uncv2.mRNAseq_RSEM_normalized_log2.txt"
                    )
                ),
                {},
                "mRNAseq_RSEM.csv",
                1,
                "gene",
                lambda x: x.lower()[:-3],
            ),
        }

        # Make single CSV files
        for dataset_name, metadata in tcga_raw_data_metadata.items():
            print(
                f"{datetime.datetime.now().strftime('%H:%M:%S')}: Processing {dataset_name} ... ",
                end="",
            )
            self.__merge_TCGA_files(metadata)
            print(f"{datetime.datetime.now().strftime('%H:%M:%S')}")

    def __merge_TCGA_files(self, dataset_metadata: TCGA_MetadataTuple):
        if not (self.tmp_dir / dataset_metadata.output_file_name).exists():
            TARGET_TREATMENT_ID = "Hybridization REF"
            dfs = []
            for file in dataset_metadata.files:
                tumor = str(file.name)[: file.name.find(".")]

                print(f"{tumor}")

                tmp = pd.read_csv(
                    file, sep="\t", header=[0], **dataset_metadata.load_csv_kwargs
                )
                tmp = tmp.T.reset_index()
                tmp.columns = tmp.iloc[0, 0:]
                tmp = tmp.iloc[dataset_metadata.header_offset :, :].reset_index(
                    drop=True
                )
                tmp[TARGET_TREATMENT_ID] = tmp[dataset_metadata.treatment_id].apply(
                    dataset_metadata.treatment_lambda
                )
                tmp["disease_code"] = tumor
                dfs.append(tmp)

            merged_df = (
                pd.concat(dfs)
                .drop(columns=["Composite Element REF"], errors="ignore")
                .replace(r"^\s*$", np.nan, regex=True)
            )
            if dataset_metadata.treatment_id != TARGET_TREATMENT_ID:
                merged_df = merged_df.drop(
                    columns=[dataset_metadata.treatment_id], errors="ignore"
                )
            del dfs
            merged_df.to_csv(
                self.tmp_dir / dataset_metadata.output_file_name, index=False
            )
            del merged_df

    def __perform_pca_write_h5(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        store = pd.HDFStore(self.processed_h5_file, "w")
        metadata_hdf = h5py.File(self.metadata_h5_file, "w")
        print("Running dimensionality reduction")

        for file_path in [
            self.tmp_dir / f
            for f in [
                "RPPA.csv",
                "miRNAseq_RPKM_log2.csv",
                "mRNAseq_RSEM.csv",
                "methylation.csv",
            ]
        ]:
            if self.custom_transformation_func is not None:
                self.custom_transformation_func(file_path, store)
            else:
                dataset_type = (
                    str(file_path.name)
                    .split("_")[0]
                    .replace(".csv", "")
                    .replace("seq", "")
                )
                index_column = "Hybridization REF"
                print(f"Performing dimensionality reduction on {file_path}")
                df = pd.read_csv(file_path)
                df = df[df[index_column].astype(str).str.startswith("tcga")]
                df = df.drop_duplicates(subset=[index_column])
                df = df[np.asarray(list(df))[df.isna().sum(axis=0) == 0]]
                df = df.set_index(index_column)

                features = sorted(
                    [
                        f
                        for f in df.columns.values
                        if f not in [index_column, "disease_code"]
                    ]
                )

                if self.vif_filter_cutoff is not None:
                    from phenonaut.transforms.preparative import (
                        RemoveHighestCorrelatedThenVIF,
                    )

                    tmp_ds = Dataset("tmp_df", df, {"features": features})
                    print("Num features in dataset:", len(tmp_ds.features))
                    # If there are too many features, then we cannot analyse all
                    # correlations and must abandon
                    if len(tmp_ds.features) <= 1000:
                        r_high_then_vif = RemoveHighestCorrelatedThenVIF(verbose=True)
                        r_high_then_vif.filter(
                            tmp_ds, vif_cutoff=self.vif_filter_cutoff, drop_columns=True
                        )
                        features = tmp_ds.features
                    df = tmp_ds.df
                sc = StandardScaler()
                df_features = sc.fit_transform(df[features])
                h5_root_group = metadata_hdf.create_group(dataset_type)
                g_scaler = h5_root_group.create_group("standardscaler")
                g_scaler.create_dataset("mean_", data=sc.mean_)
                g_scaler.create_dataset("scale_", data=sc.scale_)

                g_pca = h5_root_group.create_group("pca")

                pca = PCA(n_components=self.num_pca_dims)
                pcaspace = pca.fit_transform(df_features)
                pca_features = [f"PC{n+1}" for n in range(self.num_pca_dims)]

                pca_df = pd.DataFrame(data=pcaspace, columns=pca_features)
                pca_df["disease_code"] = df["disease_code"].astype(str).values
                pca_df[index_column] = df.index
                pca_df = pca_df[pca_df[index_column].astype(str).str.startswith("tcga")]

                pca_df.set_index(index_column, inplace=True)
                h5path = f"{dataset_type}"
                store[h5path] = pca_df

                this_pca_grp = g_pca.create_group(str(self.num_pca_dims))
                this_pca_grp.create_dataset("n_components", data=self.num_pca_dims)
                this_pca_grp.create_dataset("components", data=pca.components_)
                this_pca_grp.create_dataset("mean", data=pca.mean_)
        store["clinical_decisions"] = pd.read_csv(
            self.tmp_dir / "clinical_decisions.csv", low_memory=False
        )
        store.close()
        metadata_hdf.close()

    def get_df(self, key: str) -> pd.DataFrame:
        """Get supporting DataFrame

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
        """Get Dataset

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
        if key.startswith("/"):
            key = key[1:]
        df = store["/" + key]
        features = [f"PC{d}" for d in range(1, self.num_pca_dims + 1)]

        if self.prediction_target is not None:
            if self.prediction_target == "years_to_death":
                df = self.add_clinical_decision_data_to_df(
                    df, "days_to_death", self.prediction_target, lambda x: x / 365.25
                )
            elif self.prediction_target == "survives_1_year":
                df = self.add_clinical_decision_data_to_df(
                    df,
                    "days_to_death",
                    self.prediction_target,
                    lambda x: 1 if (x / 365.25) >= 1 else 0,
                    dtype=int,
                )
            else:
                df = self.add_clinical_decision_data_to_df(df, self.prediction_target)

        ds = Dataset(key, df, {"features": features})
        if self.prediction_target is not None:
            ds._metadata["prediction_target"] = self.prediction_target
        store.close()
        return ds

    def add_clinical_decision_data_to_df(
        self,
        df: pd.DataFrame,
        clinical_decision_column: str,
        new_df_column_name: Optional[str] = None,
        custom_func: Optional[Callable[[pd.Series], pd.Series]] = None,
        remove_incomplete_rows: bool = True,
        dtype: Type = float,
    ) -> pd.DataFrame:
        """Merge a field from TCGA clinical_decisions.

        The TCGA dataset comes with a clinical_decisions DataFrame specifying
        things like patient age, days to death etc. As years to death can be a
        prediction target, we need a way to add this to our multiomics
        DataSets/DataFrames. This function merges the clinical_decisions
        information based on the "Hybridization REF" index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to which the information will be added
        clinical_decision_column : str
            Column in clinical_decitions that will be added
        new_df_column_name : Optional[str], optional
            The new column may have a new name. If None, then the clinical_decision_column
            is used, by default None.
        custom_func : Optional[Callable[[pd.Series], pd.Series]], optional
            Optionally apply a transformation to the newly added data. It can
            be useful to pass a lambda here, which can enable a simple way to
            convert days to years, such as : 'lambda x: x/365.'. If None, then
            no transformation is applied to the new column, by default None.
        remove_incomplete_rows : bool, optional
            If True, rows containing missing data in the new column (by virtue
            of a pd.na being present) are removed, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the new column.
        """
        if new_df_column_name is None:
            new_df_column_name = clinical_decision_column

        cd_series = self.clinical_decisions_df[clinical_decision_column]
        if custom_func is not None:
            cd_series = cd_series.map(custom_func)
        df = df.join(cd_series).rename(
            columns={clinical_decision_column: new_df_column_name}
        )

        if remove_incomplete_rows:
            df = df[~df[new_df_column_name].isin([np.nan, np.inf, -np.inf])]
        if dtype is not None:
            df[new_df_column_name] = df[new_df_column_name].astype(dtype)
        return df
