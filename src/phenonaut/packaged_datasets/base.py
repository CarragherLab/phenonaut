# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import gzip
import shutil
import urllib.request
from abc import ABC, abstractclassmethod, abstractmethod
from collections.abc import Callable
from os import path
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple, Union
from urllib.error import URLError

from tqdm import tqdm


class PackagedDataset(ABC):
    """PackagedDataset base class for all downloaded Datasets

    Inherited by Phenonaut classes which supply public datasets in the same
    way that pytorch allows easy access to MNIST and FashionMNIST etc,
    Phenonaut offers classes which download and preprocess datasets like
    TCGA (The Cancer Genome Atlas and the Connectivity Map), which may
    include many different 'views' or omics-based measurements of the
    underlying cells.

    Inheriting from this class allows easy access to commonly used functions
    for checking datasets exist in directories and downloading them if not,
    in addition to more small helpful functions. Inheriting grants the
    following:

    * Getters and setters for root and raw_data_dir, properly handling the expected location of dataset files, listing available Phenonaut Dataset objects via the .keys() or ds_keys() methods, listing supporting dataframes via the df_keys() methods.
    * download and batch_download functions which simplify the download of remote public datasets.
    * processed_dataset_exists and raw_dataset_exists, which check for the presence of the processed dataset and the raw dataset, respectively.

    Inheriting classes should do the following:

    * Call super().__init__() on initialisation
    * Check if they find a saved/processed version of the packaged dataset. The CMAP and TCGA classes which inherit from this PackagedDataset process and save the datasets in an h5 file. This is optional, and any store may be used.
    * If it does not exist, download the data storing it in .raw_data_dir, process the data and store in a convenient format.
    * Register the available Phenonaut Datasets associated with this PackagedDataset.  By convention, the default/main dataset should be named 'ds'. Registration is completed by calling self.register_ds_key('ds_name'). Available Phenonaut Datasets are available by calling self.keys(), or self.ds_keys(). Phenonaut datasets may be accessed by calling get_ds, or with the ['ds_name'] notation on the PackagedDataset instance.
    * Register the available supporting dataframes associated with this PackagedDataset. Supporting dataframe names can be listed by calling df_keys() and accessed by calling get_df('df_name').
    * Classes should provide their own get_df and get_ds methods. This is enforced by this base class specifying required methods to be present through inheritance of the AbstractBaseClass.

    Parameters
    ----------
    root : Union[Path, str]
        Root directory for the dataset. The root directory should contain
        processed files, usable by Phenonaut, this means that the data has
        been downloaded and usually transformed in some manner prior to
        being put here. By convention, processed files will be put into
        this directory, but there will exist a subdirectory called
        "raw_data", within which downloaded files (possibly compressed)
        will be placed prior to preprocessing.
    raw_data_dir : Optional[Union[Path, str]], optional
        Directory in which the raw, downloaded files should be saved, also
        the location of intermediate files generated in the processing step.
        By convention, this directory lies within the root directory and has
        the default name "raw_data". It can be an absolute path in a
        different directory or filesystem by setting
        raw_data_dir_relative_to_root argument to False.
        By default Path("raw_data").
    raw_data_dir_relative_to_root : bool, optional
        As described for the raw_data_dir argument, by the "raw_data"
        directory is usually within the root directory for the dataset.
        If this argument is True, then the full path of the raw_data_dir is
        generated with root as the parent directory.  If this value is
        False, then it is taken to be an absolute path, possibly existing
        on another filesystem. By default True.
    """

    def __init__(
        self,
        root: Union[Path, str],
        raw_data_dir: Optional[Union[Path, str]] = Path("raw_data"),
        raw_data_dir_relative_to_root: bool = True,
        download: bool = True,
    ):
        """PackagedDataset base class for all downloaded Datasets

        Inherited by Phenonaut classes which supply public datasets in the same
        way that pytorch allows easy access to MNIST and FashionMNIST etc,
        Phenonaut offers classes which download and preprocess datasets like
        TCGA (The Cancer Genome Atlas and the Connectivity Map), which may
        include many different 'views' or omics-based measurements of the
        underlying cells.

        Inheriting from this class allows easy access to commonly used functions
        for checking datasets exist in directories and downloading them if not,
        in addition to more small helpful functions. Inheriting grants the
        following:

        * Getters and setters for root and raw_data_dir, properly handling the expected location of dataset files, listing available Phenonaut Dataset objects via the .keys() or ds_keys() methods, listing supporting dataframes via the df_keys() methods.
        * download and batch_download functions which simplify the download of remote public datasets.
        * processed_dataset_exists and raw_dataset_exists, which check for the presence of the processed dataset and the raw dataset, respectively.

        Inheriting classes should do the following:

        * Call super().__init__() on initialisation
        * Check if they find a saved/processed version of the packaged dataset. The CMAP and TCGA classes which inherit from this PackagedDataset process and save the datasets in an h5 file. This is optional, and any store may be used.
        * If it does not exist, download the data storing it in .raw_data_dir, process the data and store in a convenient format.
        * Register the available Phenonaut Datasets associated with this PackagedDataset.  By convention, the default/main dataset should be named 'ds'. Registration is completed by calling self.register_ds_key('ds_name'). Available Phenonaut Datasets are available by calling self.keys(), or self.ds_keys(). Phenonaut datasets may be accessed by calling get_ds, or with the ['ds_name'] notation on the PackagedDataset instance.
        * Register the available supporting dataframes associated with this PackagedDataset. Supporting dataframe names can be listed by calling df_keys() and accessed by calling get_df('df_name').
        * Classes should provide their own get_df and get_ds methods. This is enforced by this base class specifying required methods to be present through inheritance of the AbstractBaseClass.

        Parameters
        ----------
        root : Union[Path, str]
            Root directory for the dataset. The root directory should contain
            processed files, usable by Phenonaut, this means that the data has
            been downloaded and usually transformed in some manner prior to
            being put here. By convention, processed files will be put into
            this directory, but there will exist a subdirectory called
            "raw_data", within which downloaded files (possibly compressed)
            will be placed prior to preprocessing.
        raw_data_dir : Optional[Union[Path, str]], optional
            Directory in which the raw, downloaded files should be saved, also
            the location of intermediate files generated in the processing step.
            By convention, this directory lies within the root directory and has
            the default name "raw_data". It can be an absolute path in a
            different directory or filesystem by setting
            raw_data_dir_relative_to_root argument to False.
            By default Path("raw_data").
        raw_data_dir_relative_to_root : bool, optional
            As described for the raw_data_dir argument, by the "raw_data"
            directory is usually within the root directory for the dataset.
            If this argument is True, then the full path of the raw_data_dir is
            generated with root as the parent directory.  If this value is
            False, then it is taken to be an absolute path, possibly existing
            on another filesystem. By default True.
        """
        self.root = root
        self._raw_data_dir_relative_to_root = raw_data_dir_relative_to_root
        self.raw_data_dir = raw_data_dir
        self._keys_to_accessible_dfs = []
        self._keys_to_accessible_dss = []

    @property
    def root(self) -> Path:
        """Get the root directory of the dataset

        Returns
        -------
        Path
            Root directory within which the processed dataset and raw dataset
            data can be found.
        """
        return self._root

    @root.setter
    def root(self, val: Union[Path, str]):
        """Set the root directory of the dataset

        Parameters
        -------
        val :  Union[Path, str]
            Root directory within which the processed dataset and raw dataset
            data can be found.
        """
        if isinstance(val, str):
            val = Path(val)
        if not isinstance(val, Path):
            raise TypeError(f"root should be of type Path or str, but was {type(val)}")
        self._root = val

    @property
    def raw_data_dir(self) -> Path:
        """Getthe raw unprocessed data directory of the dataset

        Returns
        -------
        Path
            Directory of the raw, unprocessed dataset.
        """
        return self._raw_data_dir

    @raw_data_dir.setter
    def raw_data_dir(self, val: Union[Path, str]):
        """Set the raw unprocessed data directory of the dataset

        Parameters
        -------
        Path : Union[Path, str]
            Directory of the raw, unprocessed dataset.
        """
        if val is None:
            self._raw_data_dir = self.root
            return
        if isinstance(val, str):
            val = Path(val)
        if not isinstance(val, Path):
            raise TypeError(
                f"raw_data_dir should be of type Path or str, but was {type(val)}"
            )
        if self._raw_data_dir_relative_to_root:
            self._raw_data_dir = (self.root / val).resolve()
        else:
            self._raw_data_dir = val.resolve()

    def _download(
        self,
        source: str,
        destination: Union[Path, str],
        mkdir: bool = True,
        skip_if_exists: bool = True,
        extract: bool = False,
    ):
        """Download a remote file to local filesystem

        Parameters
        ----------
        source : str
            Remote URL of the target file.
        destination : Union[Path, str]
            Full destination path and filename.
        mkdir : bool, optional
            If True, the parent directories of the destination filenames are
            made, by default True.
        skip_if_exists: bool, optional
            If True, then skip if the file already exists, by default True.
        extract: bool, optional
            If True, then extract the downloaded archive by calling
            self.extract_archive, by default False.
        """
        if isinstance(destination, str):
            destination = Path(destination)
        if not isinstance(destination, Path):
            raise TypeError(
                f"Destination should be of type Path or str, but was {type(destination)}"
            )
        if not destination.parent.exists():
            if mkdir:
                destination.parent.mkdir(parents=True)
            else:
                raise FileNotFoundError(
                    f"Parent directories of {destination} do not existand mkdir=False, so could not make them"
                )
        if skip_if_exists and destination.exists():
            return
        urllib.request.urlretrieve(source, destination)

        if extract:
            self._extract_archive(destination)

    def _batch_download(
        self,
        tasks: List[Tuple[str, Path]],
        max_attempts=3,
        retry_pause_seconds: int = 5,
        mkdir: bool = True,
        non_critical_urls: Optional[List[str]] = None,
    ):
        """Batch download remote files to local filesystem.

        Parameters
        ----------
        tasks : List[Tuple[str, Path]]
            List of tuples, with each tuple being a download task. Each task
            consists of first a remote source location string, and then
            secondarily, a destination path where the file should be saved.
            For example:
            [("http://example1.com/remote_file1","/tmp/file1.txt),
            ("http://example2.com/remote_file2","/tmp/file2.txt)]
        max_attempts : int, optional
            Maximum number of times to retry after failed download,
            by default 3.
        retry_pause_seconds : int, optional
            Pause for this length of time in seconds before download retries, by
            default 5.
        mkdir : bool, optional
            Used as an argument to the download function used by this batch
            downloader, if true, then parent directories for the named
            destination files are made if they do not exist. By default True.
        non_critical_urls: list(str), optional
            In datasets such as TCGA, some files that should exist on the server
            are absent. If after the set number of download attempts
            (max_attempts) has been reached and the file is still not
            downloaded, no exception will be raised if the remote URL is present
            in this non_critical_urls list.  By default None.
        """
        if isinstance(non_critical_urls, str):
            non_critical_urls = [non_critical_urls]
        failed_items = []
        if max_attempts == 0:
            if all(fi[0] in non_critical_urls for fi in failed_items):
                return
            else:
                raise URLError(
                    f"Cannot make any more download attempts, {len(tasks)} tasks remain, they were:\n {tasks}. Set download=False to process what was downloaded"
                )
        for source, destination in tqdm(tasks):
            try:
                self._download(source, destination)
            except Exception as e:
                print(e)
                print(f"Failed to get {source}, will retry ")
                failed_items.append((source, destination))
        if len(failed_items) > 0:
            sleep(retry_pause_seconds)
            self._batch_download(
                failed_items,
                max_attempts=max_attempts - 1,
                retry_pause_seconds=retry_pause_seconds,
            )

    def _processed_dataset_exists(self, required_files: list[Path]) -> bool:
        """Checks if all listed files are present in the dataset root directory

        Parameters
        ----------
        required_files : List[Path]
            List of file names to check exist in dataset root directory

        Returns
        -------
        bool
            True if all files were found in root of dataset directory
        """
        return all([(self.root / file).exists() for file in required_files])

    def _raw_dataset_exists(self, required_files: list[Path]) -> bool:
        """Checks if all listed files are present in the raw_data_dir of dataset

        Parameters
        ----------
        required_files : List[Path]
            List of file names to check exist in raw_data_dir of dataset

        Returns
        -------
        bool
            True if all files were found in raw_data_dir of dataset
        """
        return all([(self.root / file).exists() for file in required_files])

    def _extract_archive(
        self,
        archive: Union[Path, str],
        destination_dir: Optional[Union[Path, str]] = None,
        keep_original: bool = False,
    ):
        """Extract archive

        Parameters
        ----------
        archive : Union[Path, str]
            Path to archive
        destination_dir : Optional[Union[Path, str]], optional
            Directory where the archive is to be extracted to.  If None, then
            the destination is the location of the archive file,
            by default None.
        keep_original : bool
            If true, the original archive is kept.  Only comes into effect if
            the archive is a .gz, excluding .tar.gz, by default False.
        """
        if isinstance(archive, str):
            archive = Path(archive)
        if destination_dir is None:
            destination_dir = archive.parent
        if not destination_dir.exists():
            destination_dir.mkdir(parents=True)
        if not str(archive).endswith(".tar.gz"):
            with gzip.open(archive, "rb") as f_in:
                with open(str(archive)[: -len(".gz")], "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            if not keep_original:
                Path(archive).unlink(missing_ok=True)
        else:
            shutil.unpack_archive(archive, destination_dir)

    def _call_if_file_missing(
        self,
        required_file: Union[Path, str],
        func: callable,
        func_kwargs: Optional[dict] = None,
        path_relative_to_root: bool = False,
    ):
        """If a file is missing, call the appropriate function

        Parameters
        ----------
        required_file : Union[Path, str]
            File that should be present.
        func : callable
            The function to be called if the file is absent.
        func_kwargs : dict
            kwargs to be used when calling the given function.
        path_relative_to_root : bool, optional
            If True, then the given path is relative to the root directory of
            the dataset. If false, then assume a full path has been supplied,
            by default False.
        """
        if path_relative_to_root:
            required_file = self.root / required_file
        if func_kwargs is None:
            func_kwargs = {}
        if not required_file.exists():
            func(**func_kwargs)

    def keys(self) -> list:
        """Get a list of Datasets contained within this PackagedDataset.

        Returns a list of Dataset names contained within this PackagedDataset
        and allows access to them via pds.['dataset_name'] - dictionary-like
        notation.

        Returns
        -------
        List
            List of keys which allow accessing pd.DataFrames belonging to this
            PackagedDataset.
        """
        return self.ds_keys()

    def __iter__(self):
        """Like a dictionary, iterating the object returns keys to Datasets

        Returns
        -------
        List
            List of dataset names
        """
        return (k for k in self.ds_keys())

    def ds_keys(self) -> list:
        """Get a list of Datasets contained within this PackagedDataset.

        Returns a list of Dataset names contained within this PackagedDataset
        and allows access to them via pds.['dataset_name'] - dictionary-like
        notation.

        Returns
        -------
        List
            List of keys which allow accessing pd.DataFrames belonging to this
            PackagedDataset
        """
        return self._keys_to_accessible_dss

    def df_keys(self) -> list:
        """Get a list of available DataFrames.

        PackagedDatasets may include DataFrames, useful in the capture of
        metadata.

        Returns
        -------
        List
            List of keys which allow accessing metadata (typically)
            pd.DataFrames belonging to this PackagedDataset.
        """
        return self._keys_to_accessible_dfs

    def register_df_key(self, key: Union[str, List[str]]) -> None:
        """Register a dataframe key with the PackagedDataset

        Packaged datasets may contain or have access to multiple pd.DataFrames
        which accompany the main dataset. In the case of the CMAP dataset, the
        main Phenonaut Dataset contains the L1000 values, along with features
        and metadata. The supporting dataframes contain information on the
        perturbation type, compound information, etc. Keys are typically the
        same as their HDF5 store key values, although this is up to the specific
        PackagedDataset implementation.

        Parameters
        ----------
        key : Union[str, List[str]]
            Short string which may be used to access the pd.DataFrame.

        Raises
        ------
        TypeError
            [description]
        """
        if isinstance(key, str):
            self._keys_to_accessible_dfs.append(key)
            return
        if isinstance(key, list):
            self._keys_to_accessible_dfs.extend(key)
            return
        raise TypeError(f"Key must be a str or list, given type was {type(key)}")

    def register_ds_key(self, key: Union[str, List[str]]) -> None:
        """Register a Phenonaut Dataset key with the PackagedDataset

        Packaged datasets may contain or have access to multiple pd.DataFrames
        from which Phenonaut Datasets can be created. These Datasets contain not
        only the pd.DataFrame containing data, but also features and additional
        metadata such as history and origin. The main Phenonaut Dataset should
        be called, by convention "ds" and additional Phenonaut Datasets given a
        descriptive name. In the case of the CMAP dataset, the main DataSet (ds)
        contains the L1000 values, feature and metadata. Supporting dataframes
        are accessible using the .get_df("name") method and contain information
        on things like the perturbation type, compound information, etc. Keys
        are typically the same as their HDF5 store key values, although this is
        up to the specific PackagedDataset implementation.

        Parameters
        ----------
        key : Union[str, List[str]]
            Short string which may be used to access the pd.DataFrame.
            When given, to the function, the name will be registered. After
            which the inheriting class should allow access to a Phenonaut
            Dataset via the __getitem__ method, allowing access to the
            Phenonaut Dataset via pdf['ds'].

        Raises
        ------
        TypeError
            Given key must be a str or list of str.
        """
        if isinstance(key, str):
            self._keys_to_accessible_dss.append(key)
            return
        if isinstance(key, list):
            self._keys_to_accessible_dss.extend(key)
            return
        raise TypeError(f"Key must be a str or list, given type was {type(key)}")

    def __getitem__(self, key):
        """Get dataset using ['ds'] notation

        Access Datasets using simple ['ds_name'] - dictionary-like notation. If
        a Dataset of that name is not present, then check if a DataFrame of that
        name is present.  If no Dataset or DataFrame is found, then an error is
        raised.

        Parameters
        ----------
        key : str
            Key of dataset to be returned, if it does not exist in ds_keys, then look in dataframes

        Returns
        -------
        Union[Dataset, pd.DataFrame, pd.Series]
            Requested Dataset of pd.DataFrame

        Raises
        ------
        KeyError
            Key not found
        """
        if key in self.ds_keys():
            return self.get_ds(key)
        if key in self.df_keys():
            return self.get_df(key)
        raise KeyError(
            f"The key '{key}' was not found in registered datsets or dataframes (.ds_keys(), or df_keys())"
        )

    @abstractmethod
    def get_df(self, key: str):
        """Abstract method - Get DataFrame

        Abstract method which all inheriting classes are required to implement
        for retrieval of DataFrames.

        Parameters
        ----------
        key : str
            Name of DataFrame
        """
        pass

    @abstractmethod
    def get_ds(self, key: str):
        """Abstract method - Get Dataset

        Abstract method which all inheriting classes are required to implement
        for retrieval of Datasets.

        Parameters
        ----------
        key : str
            Name of Dataset
        """
        pass
