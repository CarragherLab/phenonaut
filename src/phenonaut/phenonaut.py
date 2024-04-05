# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import gzip
import pickle
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterable
from copy import deepcopy
from hashlib import sha256
from itertools import combinations as itertools_combinations
from pathlib import Path
from random import random
from typing import List, Optional, Union, Literal

import pandas as pd
from pandas.errors import DataError
from sklearn.utils import Bunch

from phenonaut import data
from phenonaut.data import Dataset
from phenonaut.packaged_datasets.base import PackagedDataset
from phenonaut.utils import check_path


class Phenonaut:
    """Phenonaut object constructor

    Holds multiple datasets of different type, applys transforms, load and
    tracking operations.

    May be initialised with:

    * Phenonaut Datasets
    * Phenonaut PackageDataset
    * Scikit Bunch
    * pd.DataFrame

    by passing the object as an optional dataset argument.

    Parameters
    ----------
    dataset : Optional[Union[Dataset, list[Dataset], PackagedDataset, Bunch,
        pd.DataFrame, Path, str]], optional
        Initialise Phenonaut object with a Dataset, list of datasets, or
        PackagedDataset, by default None.
    name : str
        A name may be given to the phenonaut object. This is useful in
        naming collections of datasets. For example, The Cancer Genome Atlas
        contains 4 different views on tumors - mRNA, miRNA, methylation and
        RPPA, collectively, these 4 datasets loaded into a phenonaut object
        may be named 'TCGA' - or 'The Cancer Genome Atlas dataset'. If set
        to None, then the phenonaut object takes the name "Phenonaut data",
        however, not in the case where construction of the object occurs
        with a phenonaut packaged dataset or already named phenonaut object,
        where it takes the name of the passed object/dataset.
    kind : Optional[str]
        Instead of providing metadata, some presets are available, which make reading
        in things like DRUG-Seq easier. This argument only has an effect when reading
        in a raw data file, like CSV or H5 and directs Phenonaut to use a predefind set
        of parameters/transforms. If used as well as metadata, then the preset metadata
        dictionary from the kind argument is first loaded, then updated with anything in
        the metadata dictionary, this therefore allows overriding specific presets present
        in kind dictionaries. Available 'kind' dictionaries may be listed by examining:
        phenonaut.data.recipes.recipes.keys()
    packaged_dataset_name_filter : Optional[Union[list[str], str]], optional
        If a PackagedDataset is supplied for the data argument, then import
        only datasets from it named in the name_filter argument. If None,
        then all PackagedDataset datasets are imported.  Can be a single
        string or list of strings. If None, and PackagedDataset is supplied,
        then all Datasets are loaded. Has no effect if data is not a
        PackagedDataset, by default None.
    metadata : Optional[Union[dict, list[dict]]]
        Used when a pandas DataFrame is passed to the constructor of the
        phenonaut object. Metadata typically contains features or
        feature_prefix keys telling Phenonaut which columns should be
        treated as Dataset features. Can also be a list of metadata
        dicitonaries if a list of pandas DataFrames are supplied to the
        constructor. Has no effect if the type of dataset passed is not a
        pandas DataFrame or list of pandas DataFrames. If a list of pandas
        DataFrames is passed to data but only one metadata dictionary is
        given, then this dictionary is applied to all DataFrames. By default {}.
    features : Optional[list[str]] = None
        May be used as a shortcut to including features in the metadata dictionary.
        Only used if the metadata is a dict and does not contain a features key.
    dataframe_name : Optional[Union[dict, list[dict]]]
        Used when a pandas DataFrame, or str, or Path to a CSV file is passed to
        the constructor of the phenonaut object. Optional name to give to the
        dataset object constructed from the pandas DataFrame. If multiple
        DataFrames are given in a list, then this dataframe_name argument can be
        a list of strings as names to assign to the new Dataset objects.
    init_hash : Optional[Union[str, bytes]]
        Cryptographic hashing within Phenonaut can be initialised with a
        starting/seed hash. This is useful in the creation of
        blockchain-like chains of hashes. In environments where timestamping
        is unavailable, hashes may be published and then used as input to
        subsequent experiments. Building up a provable chain along the way.
        By default None, implying an empty bytes array.
    """

    def __init__(
        self,
        dataset: Optional[
            Union[
                Dataset, list[Dataset], PackagedDataset, Bunch, pd.DataFrame, Path, str
            ]
        ] = None,
        name: str = "Phenonaut object",
        kind: Optional[str] = None,
        packaged_dataset_name_filter: Optional[Union[list[str], str]] = None,
        metadata: Optional[Union[dict, list[dict]]] = {},
        features: Optional[list[str]] = None,
        dataframe_name: Optional[Union[str, list[str]]] = None,
        init_hash: Optional[Union[str, bytes]] = None,
    ):
        metadata = deepcopy(metadata)
        if (
            isinstance(metadata, dict)
            and "features" not in metadata
            and features is not None
        ):
            metadata["features"] = features.copy()
        if init_hash is None:
            self.init_hash = b""
        else:
            if isinstance(init_hash, bytes):
                self.init_hash = init_hash
            elif isinstance(init_hash, str):
                self.init_hash = init_hash.encode("utf-8")
            else:
                raise ValueError(
                    f"The given argument init_hash was not a bytes array or a string"
                )
        self.datasets = []
        self.name = name
        if dataset is None:
            return
        if isinstance(dataset, Phenonaut):
            for ds in dataset.datasets:
                self.datasets.append(ds.copy())
            if name is None and dataset.name is not None:
                self.name = dataset.name
            return

        # If scikit bunch
        if isinstance(dataset, Bunch):
            # Check required keys are in Bunch
            required_bunch_keys = ["data", "feature_names", "target"]
            missing_keys = set(required_bunch_keys) - set(dataset.keys())
            if len(missing_keys) > 0:
                raise ValueError(
                    f"The supplied bunch did not contain the required keys ({required_bunch_keys}), containing only: {dataset.keys()}, missing keys were: {missing_keys}"
                )
            # Construct pd.DataFrame from the bunch.
            df = pd.concat(
                [
                    pd.DataFrame(dataset.data, columns=dataset.feature_names),
                    pd.DataFrame(dataset.target, columns=["target"]),
                ],
                axis=1,
            )
            # Append to datasets, setting features and name.
            if dataframe_name is None:
                self.datasets.append(
                    Dataset(
                        dataset_name="sklearnBunch",
                        input_file_path_or_df=df,
                        metadata={"features": dataset.feature_names},
                    )
                )
            else:
                self.datasets.append(
                    Dataset(
                        dataset_name=dataframe_name,
                        input_file_path_or_df=df,
                        metadata={"features": dataset.feature_names},
                    )
                )

            if name is not None:
                self.name = name
            return

        # If Path or string, then read it in
        if isinstance(dataset, (Path, str)):
            if isinstance(dataset, str):
                dataset = Path(dataset)
            if dataframe_name is None:
                dataframe_name = str(dataset)
            dataset = Dataset(
                dataset_name=dataframe_name,
                input_file_path_or_df=dataset,
                metadata=metadata,
                kind=kind,
            )

        # If its a Dataset, or list of datasets, directly add it to self.datasets.
        if isinstance(dataset, (Dataset, pd.DataFrame)):
            dataset = [dataset]
        if isinstance(dataset, list):
            ds_types = list(set(map(type, dataset)))

            if len(ds_types) > 1:
                raise ValueError(
                    f"If passing a list for the data argument, all elements must be of the same type, they were {list(map(type, dataset))}"
                )
            if ds_types[0] is not pd.DataFrame and ds_types[0] is not Dataset:
                raise ValueError(
                    f"List containing {ds_types} passed, but lists to constructor must contain phenonaut Datasets or pandas DataFrames"
                )

            if ds_types[0] == Dataset:
                self.datasets.extend(dataset)
                return

            if ds_types[0] == pd.DataFrame:
                if isinstance(dataframe_name, str):
                    dataframe_name = [dataframe_name] * len(dataset)
                if dataframe_name is None:
                    dataframe_name = ["Dataset from pd.DataFrame"] * len(dataset)
                if len(dataframe_name) != len(dataset):
                    raise ValueError(
                        f"dataframe_name contained {len(dataframe_name)} name(s), but dataset contained {len(dataset)} pd.DataFrames"
                    )
                if metadata is None:
                    metadata = {}
                if isinstance(metadata, dict):
                    metadata = [metadata.copy() for dm in range(len(dataset))]
                if len(metadata) != len(dataset):
                    raise ValueError(
                        f"dataset_metadata contained {len(metadata)} metadata dictionaries, but dataset contained {len(dataset)} pd.DataFrames"
                    )
                for ds_name, df, ds_metadata in zip(dataframe_name, dataset, metadata):
                    self.datasets.append(
                        Dataset(
                            dataset_name=ds_name,
                            input_file_path_or_df=df,
                            metadata=ds_metadata,
                        )
                    )
                return

            if name is not None:
                self.name = name
            return
        # If a packaged dataset has been passed, then load it
        if issubclass(type(dataset), PackagedDataset):
            if isinstance(packaged_dataset_name_filter, str):
                packaged_dataset_name_filter = [packaged_dataset_name_filter]
            for key in dataset.keys():
                if packaged_dataset_name_filter is None:
                    self.datasets.append(dataset[key])
                else:
                    if key in packaged_dataset_name_filter:
                        self.datasets.append(dataset[key])
            if name is None:
                if dataset.name is not None:
                    self.name = dataset.name
            else:
                self.name = name
            return
        raise ValueError(
            f"Expected data to be a Dataset, pandas DataFrame, list of Datasets, list of pandas DataFrames, PackagedDataset, sklearn Bunch, or Phenonaut object, however, it was {type(dataset)}"
        )

    def __getitem__(self, dataset: Union[str, int, list[str], list[int]]) -> Dataset:
        """Allow dictionary-like access to Datasets

        Allows simple access to a Phenonaut object's Datasets using their index
        within phe.datasets, or by the name of the Dataset (instance.name)

        Parameters
        ----------
        dataset : Union[str, int]
            Can be the name of a dataset, or an index within Phenonaut.datasets.
            It can also be a list of Dataset names or indexes in order to return
            a tuple of requested Datasets.

        Returns
        -------
        Dataset
            The requested Dataset

        Raises
        ------
        KeyError
            Dataset with the supplied name was not found
        ValueError
            Invalid dataset index
        """
        if isinstance(dataset, str):
            if dataset not in self.keys():
                raise KeyError(
                    f"'{dataset}' not found as a name in datasets ({self.keys()})"
                )
            return self.datasets[self.keys().index(dataset)]
        elif isinstance(dataset, int):
            return self.datasets[dataset]
        elif isinstance(dataset, (list, tuple)):
            return [self[ds] for ds in dataset]
        else:
            raise ValueError(
                f"Attempting to get dataset, but {type(dataset)} was supplied. It should be str name, int giving the dataset location in the Phenonaut.datasets list, or a list/tuple of names/indexes to return multiple datasets"
            )

    def __setitem__(self, name: Union[str, int], dataset: Dataset) -> None:
        """Allows asignment of Datasets to Phenonaut object by name

        Parameters
        ----------
        name : str
            Name of dataset
        dataset : Dataset
            New Dataset to be stored in Phenonaut object
        """
        if isinstance(name, int):
            self.datasets[name] = dataset
        else:
            dataset.name = name
            if name in self.keys():
                name = self.get_dataset_index_from_name(name)
                self.datasets[name] = dataset
            else:
                self.datasets.append(dataset)

    def __delitem__(self, key: Union[str, int]):
        """Allows the deletion of Datasets stored in a Phenonaut object

        Allows the simple deletion of Datasets stored within a Phenonaut object
        by name or index.  Allows the simple notation:
        del phe['my_dataset']
        to delete a dataset called 'my_dataset', alternatively, we may use
        indexing like
        del phe[-1]
        to remove the last added Dataset to the Phenonaut object.

        Parameters
        ----------
        key : Union[str, int]
            Name of the dataset or the index of the Dataset to be removed.

        Raises
        ------
        KeyError
            Dataset of that name not found

        """
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(
                    f"'{key}' not found as a name in datasets ({self.keys()})"
                )
            del self.datasets[self.keys().index(key)]
            return
        elif isinstance(key, int):
            del self.datasets[key]
            return

    def __iter__(self):
        """Like a dictionary, iterating a Phenonaut object returns dataset keys

        Returns
        -------
        List
            List of dataset names
        """
        return (k for k in self.keys())

    @property
    def data(self) -> Dataset:
        """Return the data of the highest index in phenonaut.datasets

        Calling phe.data is the same as calling phe.ds.data

        Returns
        -------
        np.ndarray
            Last added/highest indexed Dataset features

        Raises
        ------
        DataError
            No datasets loaded
        """
        if self.datasets is None or self.datasets == []:
            raise DataError("Attempt to get data, but no dataset loaded")
        return self.datasets[-1].data

    @property
    def ds(self) -> Dataset:
        """Return the dataset with the highest index in phenonaut.datasets

        Returns
        -------
        Dataset
            Last added/highest indexed Dataset

        Raises
        ------
        DataError
            No datasets loaded
        """
        if self.datasets is None or self.datasets == []:
            raise DataError("Attempt to get dataset, but no dataset loaded")
        return self.datasets[-1]

    @property
    def df(self) -> pd.DataFrame:
        """Return the pd.DataFrame of the last added/highest indexed Dataset

        Returns the internal pd.Dataframe of the Dataset contained within the
        Phenonaut instance's datasets list.

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return self.ds.df

    def keys(self) -> list[str]:
        """Return a list of all dataset names

        Returns
        -------
        list(str)
            List of dataset names, empty list if no datasets are loaded.
        """
        if self.datasets is None or self.datasets == []:
            return []
        return [ds.name for ds in self.datasets]

    def get_hash_dictionary(self) -> dict:
        """Returns dictionary containing SHA256 hashes

        Returns a dictionary of base64 encoded UTF-8 strings representing the
        SHA256 hashes of datasets (along with names), combined datasets, and
        the Phenonaut object (including name).

        Returns
        -------
        dict
            Dictionary of base64 encoded SHA256 representing datasets and
            the Phenonaut object which created them.
        """
        ds_hashes = {
            f"{idx}:{ds.name}": b64encode(ds.sha256.digest()).decode("utf-8")
            for idx, ds in enumerate(self.datasets)
        }
        phenonaut_object_hash = sha256(self.init_hash)
        combined_ds_hash = sha256(self.init_hash)

        for ds in self.datasets:
            combined_ds_hash.update(ds.sha256.digest())
            phenonaut_object_hash.update(ds.sha256.digest())
        phenonaut_object_hash.update(self.name.encode("utf-8"))

        return {
            "datasets": ds_hashes,
            "combined_datasets_hash": b64encode(combined_ds_hash.digest()).decode(
                "utf-8"
            ),
            "phenonaut_object_name": self.name,
            "phenonaut_object_hash": b64encode(phenonaut_object_hash.digest()).decode(
                "utf-8"
            ),
        }

    def __delete__(self) -> None:
        """Deconstructor for Phenonaut object

        Upon deconstruction of a Phenonaut object, a hash dictionary is
        retrieved and printed to standard output.

        """
        print(
            "Phenonaut object destroyed, hash information\n"
            "--------------------------------------------\n"
        )
        for k, v in self.get_hash_dictionary().items():
            print(f"{k} : {v}")

    def load_dataset(
        self,
        dataset_name: str,
        input_file_path: Union[Path, str],
        metadata: dict = None,
        h5_key: Optional[str] = None,
        features: Optional[list[str]] = None,
    ):
        """Load a dataset from a CSV, optionally suppying metadata and a name

        Parameters
        ----------
        dataset_name : str
            Name to be assigned to the dataset
        input_file_path : Union[Path, str]
            CSV/TSV/H5 file location
        metadata : dict, optional
            Metadata dictionary describing the CSV data format, by default None
        h5_key : Optional[str]
            If input_file_path is an h5 file, then a key to access the target
            DataFrame must be supplied.
        features : Optional[list[str]]
            Optionally supply a list of features here. If None, then the
            features/feature finding related keys in metadata are used. You may
            also explicitly supply an empty list to explicitly specify that the
            dataset has no features. This is not recommended.
        """
        if dataset_name is None and metadata is not None:
            dataset_name = metadata.pop(
                "dataset_name", f"Dataset {len(self.datasets)+1}"
            )
        if features != None:
            metadata["features"] = features
        self.datasets.append(
            Dataset(
                dataset_name=dataset_name,
                input_file_path_or_df=input_file_path,
                metadata=metadata,
            )
        )

    def combine_datasets(
        self,
        dataset_ids_to_combine: Optional[Union[list[str], list[int]]] = None,
        new_name: Optional[str] = None,
        features: list = None,
    ):
        """Combine multiple datasets into a single dataset

        Often, large datasets are split across multiple CSV files. For example,
        one CSV file per screening plate.  In this instance, it is prudent to
        combine the datasets into one.

        Parameters
        ----------
        dataset_ids_to_combine : Optional[Union[list[str], list[int]]]
            List of dataset indexes, or list of names of datasets to combine.
            For example, after loading in 2 datasets, the list [0,1] would be given,
            or a list of their names resulting in a new third dataset in datasets[2].
            If None, then all present datasets are used for the merge. By default, None.
        new_name :new_name:Optional[str]
            Name that should be given to the newly created dataset. If None, then it is
            assigned as: "Combined_dataset from datasets[DS_INDEX_LIST]", where DS_INDEX_LIST
            is a list of combined dataset indexes.
        features : list, optional
            List of new features which should used by the newly created dataset
            if None, then features of combined datasets are used. By default
            None.

        Raises
        ------
        DataError
            Error raised if the combined datasets do not have the same features.
        """
        print(dataset_ids_to_combine)
        if dataset_ids_to_combine == []:
            raise ValueError("Empty list supplied as indexes of datasets to combine")
        if dataset_ids_to_combine is None:
            dataset_ids_to_combine = [idx for idx in range(len(self.datasets))]

        dataframes_to_combine = [self[i].df for i in dataset_ids_to_combine]
        datasets_to_combine = [self[i] for i in dataset_ids_to_combine]
        new_metadata = {}
        for ds in datasets_to_combine:
            for k, v in ds._metadata.items():
                new_metadata[k] = v
        new_df = pd.concat(dataframes_to_combine, axis=0).reset_index()

        if features is None:
            features = datasets_to_combine[0].features
            for ds in datasets_to_combine[1:]:
                if ds.features != features:
                    raise DataError(
                        f"Different features found when combining datasets\n{features}\nvs\n{ds.features}"
                    )
        if new_name is None:
            new_name = f"Combined_dataset from datasets[{dataset_ids_to_combine}]"
        self.datasets.append(
            Dataset(dataset_name=new_name, input_file_path_or_df=None, metadata=None)
        )
        self.datasets[-1].df = new_df
        self.datasets[-1]._metadata = new_metadata
        self.datasets[-1].features = (
            features,
            f"Combined datasets at dataset indexes: {dataset_ids_to_combine}",
        )

    def get_dataset_names(self) -> list[str]:
        """Get a list of dataset names

        Returns
        -------
        list[str]
            List containing the names of datasets within this Phenonaut object.
        """
        return self.keys()

    def get_dataset_index_from_name(
        self, name: Union[str, list[str], tuple[str]]
    ) -> Union[int, list[int]]:
        """Get dataset index from name

        Given the name of a dataset, return the index of it in datasets list.
        Accepts single string query, or a list/tuple of names to return lists
        of indices.

        Parameters
        ----------
        name : Union[str, list[str], tuple[str]]
            If string, then this is the dataset name being searched for. Its
            index in the datasets list will be returned. If a list or tuple
            of names, then the index of each is searched and an index list
            returned.

        Returns
        -------
        Union[int, list[int]]
            If name argument is a string, then the dataset index is returned.
            If name argument is a list or tuple, then a list of indexes for
            each dataset name index is returned.

        Raises
        ------
        ValueError
            Error raised if no datasets were found to match a requested name.
        """
        if isinstance(name, str):
            return self.keys().index(name)
        elif isinstance(name, (list, tuple)):
            return [self.keys().index(n) for n in name]

    def clone_dataset(
        self,
        existing_dataset: Union[Dataset, str, int],
        new_dataset_name: str,
        overwrite_existing: bool = False,
    ) -> None:
        """Clone a dataset into a new dataset

        Parameters
        ----------
        existing_dataset : Union[Dataset, str, int]
            The name or index of an existing Phenonaut Dataset held in the
            Phenonaut object. Can also be a Phenonaut.Dataset object passed
            directly.
        new_dataset_name : str
            A name for the new cloned Dataset.
        overwrite_existing : bool, optional
            If a dataset by this name exists, then overwrite it, otherwise, an
            exception is raised, by default False.

        Raises
        ------
        ValueError
            Dataset by the name given already exists and overwrite_existing was
            False.
        ValueError
            The existing_dataset argument should be a str, int or
            Phenonaut.Dataset.
        """
        if new_dataset_name in self.keys():
            if not overwrite_existing:
                raise ValueError(
                    f"new_dataset_name ({new_dataset_name}) is already the name of an "
                    "existing dataset, and overwrite_existing was False"
                )
        if isinstance(existing_dataset, (str, int)):
            new_ds = self[existing_dataset].copy()
        elif isinstance(existing_dataset, Dataset):
            new_ds = existing_dataset.copy()
        else:
            raise ValueError(
                f"The existing_dataset argument should be a string (giving the name of an "
                "existing dataset), an int (giving the index of an existing dataset), or "
                " a phenonaut.Dataset object"
            )
        new_ds.name = new_dataset_name

        if new_dataset_name in self.keys():
            self.datasets[self.keys().index(new_dataset_name)] = new_ds
        else:
            self.datasets.append(new_ds)

    def new_dataset_from_query(
        self,
        name,
        query: str,
        query_dataset_name_or_index: Union[int, str] = -1,
        raise_error_on_empty: bool = True,
        overwrite_existing: bool = False,
    ):
        """Add new dataset through a pandas query of existing dataset

        Parameters
        ----------
        query : str
            The pandas query used to select the new dataset
        name : str
            A name for the new dataset
        query_dataset_name_or_index : Union[int, str], optional
            The dataset to be queried, can be an int index, or the name of an
            existing dataset. List indexing can also be used, such that -1 uses
            the last dataset in Phenonaut.datasets list, by default -1.
        raise_error_on_empty : bool
            Raise a ValueError is the query returns an empty dataset.  By
            default True.
        overwrite_existing : bool
            If a dataset already exists with the name given in the name
            argument, then this argument can be used to overwrite it, by
            default False.
        """
        new_ds = self[query_dataset_name_or_index].get_ds_from_query(name, query)
        if raise_error_on_empty and new_ds.df.shape[0] == 0:
            raise ValueError("Query used to create a new dataset returned no results")
        if name in self.keys():
            if overwrite_existing:
                self[name] = new_ds
            else:
                raise ValueError(
                    "Dataset with name given in the name argument already exists, "
                    "and the overwrite_existing argument was False"
                )
        else:
            self[name] = new_ds

    def add_well_id(
        self,
        numerical_column_name: str = "COLUMN",
        numerical_row_name: str = "ROW",
        plate_type: int = 384,
        new_well_column_name: str = "Well",
        add_empty_wells: bool = False,
        plate_barcode_column: str = None,
        no_sort: bool = False,
    ):
        """Add standard well IDs - such as A1, A2, etc to ALL loaded Datasets.

        If a dataset contains numerical row and column names, then they may be
        translated into standard letter-number well IDs. This is applied to all
        loaded Datasets.  If you wish only one to be annotated, then call
        add_well_id on that individual dataset.

        Parameters
        ----------
        numerical_column_name : str, optional
            Name of column containing numeric column number, by default
            "COLUMN".
        numerical_row_name : str, optional
            Name of column containing numeric column number, by default "ROW".
        plate_type : int, optional
            Plate type - note, at present, only 384 well plate format is
            supported, by default 384.
        new_well_column_name : str, optional
            Name of new column containing letter-number well ID, by default
            "Well".
        add_empty_wells : bool, optional
            Should all wells from a plate be inserted, even when missing from
            the data, by default False.
        plate_barcode_column : str, optional
            Multiple plates may be in a dataset, this column contains their
            unique ID, by default None.
        no_sort : bool, optional
            Do not resort the dataset by well ID, by default False
        """
        for ds in self.datasets:
            ds.add_well_id(
                numerical_column_name=numerical_column_name,
                numerical_row_name=numerical_row_name,
                plate_type=plate_type,
                new_well_column_name=new_well_column_name,
                add_empty_wells=add_empty_wells,
                plate_barcode_column=plate_barcode_column,
                no_sort=no_sort,
            )

    def subtract_median_perturbation(
        self,
        perturbation_label: str,
        per_column_name: Optional[str] = None,
        new_features_prefix: str = "SMP_",
    ):
        r"""Subtract the median perturbation from all features for all datasets.

        Useful for normalisation within a well/plate format. The median feature
        may be identified through the per_column_name variable, and perturbation
        label. Newly generated features may have their prefixes controled via
        the new_features_prefix argument.

        Parameters
        ----------
        perturbation_label : str
            The perturbation label which should be used to calculate the median
        per_column_name : Optional[str], optional
            The perturbation column name.  This is optional and can be None, as
            the Dataset may already have perturbation column set. By
            default, None.
        new_features_prefix : str
            Prefix for new features, each with the median perturbation
            subtracted. By default 'SMP\_' (for subtracted median perturbation).

        """
        for ds in self.datasets:
            ds.subtract_median_perturbation(
                perturbation_label, per_column_name, new_features_prefix
            )

    def get_dataset_combinations(
        self,
        min_datasets: Optional[int] = None,
        max_datasets: Optional[int] = None,
        return_indexes: bool = False,
    ):
        """Get tuple of all dataset name combinations, picking 1 to n datasets

        The function to return all combinations from 1 to n dataset names in
        combination, where n is the number of loaded datasets. This is useful
        in multiomics settings where we test A,B, and C alone, A&B, A&C, B&C,
        and finally A&B&C.

        A limit on the number of datasets in a combination can be imposed
        using the max_datasets argument. In the example above with datasets
        A, B and C, passing max_datasets=2 would return the following tuple:
        ((A), (B), (C), (A, B), (A, C), (B, C))
        leaving out the tripple length combination (A, B, C).

        Similarly, the argument min_datasets can specify a lower limit on
        the number of dataset combinations.

        Using the example with datasets A, B, and C, and setting min_datasets=2
        with no limit on max_datasets on the above example would return the
        following tuple:
        ((A, B), (A, C), (B, C), (A, B, C))

        If return_indexes is True, then the indexes of Datasets are returned.
        As directly above, datasets A, B, and C, setting min_datasets=2
        with no limit on max_datasets and passing return_indexes=True would
        return the following tuple:
        ((0, 1), (0, 2), (1, 2), (0, 1, 2))

        Parameters
        ----------
        min_datasets : Optional[int], optional
            Minimum number of datasets to return in a combination. If None,
            then it behaves as if 1 is given, by default None.
        max_datasets : Optional[int], optional
            Maximum number of datasets to return in a combination. If None,
            then it behaves as if len(datasets) is given, by default None.
        return_indexes : bool
            Return indexes of Datasets, instead of their names,
            by default False.
        """
        ds_names = None
        if return_indexes:
            ds_names = range(len(self.datasets))
        else:
            ds_names = self.keys()

        if max_datasets is None:
            max_datasets = len(ds_names)
        if min_datasets is None:
            min_datasets = 1
        if min_datasets > max_datasets:
            raise ValueError("min_datasets cannot be larger than max_datasets")

        return tuple(
            c
            for i in range(min_datasets, max_datasets + 1)
            for c in itertools_combinations(ds_names, i)
        )

    def aggregate_dataset(
        self,
        composite_identifier_columns: list[str],
        datasets: Union[Iterable[int], Iterable[str], int, str] = -1,
        new_names_or_prefix: Union[list[str], tuple[str], str] = "Aggregated_",
        inplace: bool = False,
        transformation_lookup: dict[str, Union[Callable, str]] = None,
        tranformation_lookup_default_value: Union[str, Callable] = "mean",
    ):
        r"""Aggregate multiple or single phenonaut dataset rows

        If we have a Phenonaut object containing data derived from 2 fields of
        view from a microscopy image, a sensible approach is averaging features.
        If we have the DataFrame below, we may merge FOV 1 and FOV 2, taking the
        mean of all features.  As strings such as filenames should be kept, they
        are concatenated together, separated by a comma, unless the strings are
        the same, in which case just one is used.

        Here we test a df as follows:

        === ====== ======= ====== ====== ====== ========= ===
        ROW COLUMN BARCODE feat_1 feat_2 feat_3 filename  FOV
        === ====== ======= ====== ====== ====== ========= ===
        1   1      Plate1  1.2    1.2    1.3    FileA.png 1
        1   1      Plate1  1.3    1.4    1.5    FileB.png 2
        1   1      Plate2  5.2    5.1    5      FileC.png 1
        1   1      Plate2  6.2    6.1    6.8    FileD.png 2
        1   2      Plate1  0.1    0.2    0.3    FileE.png 1
        1   2      Plate1  0.2    0.2    0.38   FileF.png 2
        === ====== ======= ====== ====== ====== ========= ===

        With just this loaded into a phenonaut object, we can call:

        phe.aggregate_dataset(['ROW','COLUMN','BARCODE'])

        Will merge and proeduce another, secondary dataset in the phe object
        containing:

        === ====== ======= ======= ======= =======  =================== ====
        ROW COLUMN BARCODE  feat_1  feat_2  feat_3            filename   FOV
        === ====== ======= ======= ======= =======  =================== ====
        1       1   Plate1    1.25     1.3    1.40  fileA.png,FileB.png  1.5
        1       1   Plate2    5.70     5.6    5.90  FileC.png,FileD.png  1.5
        1       2   Plate1    0.15     0.2    0.34  FileF.png,fileE.png  1.5
        === ====== ======= ======= ======= =======  =================== ====

        if inplace=True is passed in the call to aggregate_dataset, then the
        phenonaut object will contain just one dataset, the new aggregated
        dataset.

        Parameters
        ----------
        composite_identifier_columns : list[str]
            If a biochemical assay evaluated through imaging is identified by a
            row, column, and barcode (for the plate) but multiple images taken
            from a well, then these multiple fields of view can be merged,
            creating averaged features using row, column and barcode as the
            composite identifier on which to merge fields of view.
        datasets : Union[list[int], list[str], int, str]
            Which datasets to apply the aggregation to.  If int, then the
            dataset with that index undergoes aggregation. If a string, then the
            dataset with that name undergoes aggregation. It may also be a list
            or tuple of mixed int and string types, with ints specifying dataset
            indexes and strings indicating dataset names. By default, this value is -1,
            indicating that the last added dataset should undergo aggregation.
        new_names_or_prefix : Union[list[str], tuple[str], str]
            If a list or tuple of strings is passed, then use them as the names
            for the new datasets after aggregation. If a single string is
            passed, then use this as a prefix for the new dataset.
            By default "Aggregated\_".
        inplace : bool
            Perform the aggregation in place, overwriting the origianl
            dataframes. By default False.
        transformation_lookup : dict[str,Union[Callable, str]]
            Dictionary mapping data types to aggregations. When None, it is as
            if the dictionary:
            {np.dtype("O"): lambda x: ",".join([f"{item}" for item in set(x)])}
            was provided, concatenating strings together (separated by a comma)
            if they are different and just using one if they are the same across
            rows. If a type not present in the dictionary is encountered (such
            as int, or float) in the above example, then the default specified
            by transformation_lookup_default_value is returned.
            By default, None.
        tranformation_lookup_default_value : Union[str, Callable]
            Transformation to apply if the data type is not found in the
            transformation_lookup_dictionary, can be a callable or string to
            pandas defined string to function shortcut mappings.
            By default "mean".

        """
        if isinstance(composite_identifier_columns, str):
            composite_identifier_columns = [composite_identifier_columns]

        if isinstance(datasets, (int, str)):
            datasets = [datasets]
        datasets = list(
            datasets
        )  # In case generator is consumed by new_names_or_prefix below.

        if isinstance(new_names_or_prefix, str):
            new_names_or_prefix = [
                f"{new_names_or_prefix}{self[idx].name}" for idx in datasets
            ]

        if len(new_names_or_prefix) != len(datasets):
            raise ValueError(
                f"Expected new_names_or_prefix to have the same number of elements as number datasets being worked on. Was {len(new_names_or_prefix)}, when it should be ({len(datasets)})"
            )
        for dsi, new_ds_name in zip(datasets, new_names_or_prefix):
            new_ds = self[dsi].new_aggregated_dataset(
                composite_identifier_columns,
                new_dataset_name=new_ds_name,
                transformation_lookup=transformation_lookup,
                tranformation_lookup_default_value=tranformation_lookup_default_value,
            )
            if inplace:
                self[dsi] = new_ds
            else:
                self.datasets.append(new_ds)

    def save(
        self, output_filename: Union[str, Path], overwrite_existing: bool = False
    ) -> None:
        """Save Phenonaut object and contained Data to a pickle

        Writes a gzipped Python pickle file. If no compression, or another
        compressison format is required, then the user should use a custom
        pickle.dump and not rely on this helper function.

        Parameters
        ----------
        output_filename : Union[str, Path]
            Output filename for the gzipped pickle
        overwrite_existing : bool, optional
            If True and the file exists, overwrite it. By default False.
        """
        for ds in self.datasets:
            ds.sha256 = b64encode(ds.sha256.digest()).decode("utf-8")
        output_filename = check_path(output_filename)
        if output_filename.exists() and not overwrite_existing:
            raise FileExistsError(
                f"{output_filename} exists, and overwrite_existing was False"
            )
        self.last_saved_pickle_filename = output_filename
        with gzip.open(output_filename, "wb") as f:
            pickle.dump(self, f)
        for ds in self.datasets:
            ds.sha256 = sha256(ds.sha256.encode("utf-8"))

    def revert(self) -> None:
        """Revert a Phenonaut object that was recently saved to it's previous state

        Upon calling save on a Phenonaut object, the record stores the output file location,
        allowing a quick was to revert changes made to a file by calling .revert(). This
        returns the object to it's saved state.

        Raises
        ------
        FileNotFoundError
            File not found, Phenonaut object has never been written out.

        """
        if hasattr(self, "last_saved_pickle_filename"):
            if not self.last_saved_pickle_filename.exists():
                raise FileNotFoundError(
                    f"Could not find file {self.last_saved_pickle_filename}"
                )
            with gzip.open(self.last_saved_pickle_filename, "rb") as f:
                phe = pickle.load(f)
                for ds in phe.datasets:
                    ds.sha256 = sha256(ds.sha256.encode("utf-8"))
                self.__dict__ = phe.__dict__
        else:
            raise FileNotFoundError(
                "Called revert on a Phenonaut object which has never been saved.  Save it first, before calling revert."
            )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Phenonaut":
        """Class method to load a compressed Phenonaut object

        Loads a gzipped Python pickle containing a Phenonaut object

        Parameters
        ----------
        filepath : Union[str, Path]
            Location of gzipped Phenonaut object pickle

        Returns
        -------
        phenonaut.Phenonaut
            Loaded Phenonaut object.

        Raises
        ------
        FileNotFoundError
            File not found, unable to load pickled Phenonaut object.
        """
        filepath = check_path(filepath, make_parents=False)
        if not filepath.exists():
            raise FileNotFoundError(f"Could not find file {filepath}")
        with gzip.open(filepath, "rb") as f:
            phe = pickle.load(f)
            if isinstance(phe, Dataset):
                phe = Phenonaut(phe)
            for ds in phe.datasets:
                ds.sha256 = sha256(ds.sha256.encode("utf-8"))
            return phe

    def get_df_features_perturbation_column(
        self, ds_index=-1, quiet: bool = False
    ) -> tuple[pd.DataFrame, list[str], Union[str, None]]:
        """Helper function to obtain DataFrame, features and perturbation column name.

        Some Phenonaut functions allow passing of a Phenonaut object, or DataSet. They
        then access the underlying pd.DataFrame for calculations. This helper function
        is present on Phenonaut objects and Dataset objects, allowing more concise
        code and less replication when obtaining the underlying data. If multiple
        Datasets are present, then the last added Dataset is used to obtain data, but
        this behaviour can be changed by passing the ds_index argument.

        Parameters
        ----------
        ds_index : int, optional
            Index of the Dataset to be returned. By default -1, which uses the
            last added Dataset.
        quiet : bool
            When checking if perturbation is set, check without inducing a
            warning if it is None.

        Returns
        -------
        tuple[pd.DataFrame, list[str], str]
            Tuple containing the Dataframe, a list of features and the
            perturbation column name.
        """
        return (
            self[ds_index].df,
            self[ds_index].features,
            self[ds_index]._metadata.get("perturbation_column", None)
            if quiet
            else self[ds_index].perturbation_column,
        )

    def groupby_datasets(
        self, by: Union[str, List[str]], ds_index=-1, remove_original=True
    ):
        """Perform a groupby operation on a dataset

        Akin to performing a groupby operation on a pd.DataFrame, this splits a dataset
        by column(s), optionally keeping or removing (by default) the original.

        Parameters
        ----------
        by : Union[str, list]
            Columns in the dataset's DataFrames which should be used for grouping
        ds_index : int, optional
            Index of the Dataset to be returned. By default -1, which uses the
            last added Dataset.
        remove_original : bool, optional
            If True, then the original split dataset is deleted after splitting
        """
        target_ds = self[ds_index]
        if remove_original:
            del self[ds_index]
        self.datasets.extend(target_ds.groupby(by))

    def merge_datasets(
        self,
        datasets: Union[List[Dataset], List[int], Literal["all"]] = "all",
        new_dataset_name: Optional[str] = "Merged Dataset",
        return_merged: bool = False,
        remove_merged: Optional[bool] = True,
    ):
        """Merge datasets

        After performing a groupby operation on Phenonaut.Dataset objects, a list of
        datasets may be merged into a single Dataset using this method.

        Parameters
        ----------
        datasets : Union[List[Dataset], List[int], List[str]]
            Datasets which should be groupbed together. May be a list of Datasets, in
            which case these are merged together and inserted into the Phenonaut object,
            or a list of integers or dataset names which will be used to look up
            datasets in the current Phenonaut object.  Mixing of ints and dataset string
            identifiers is acceptable but mixing of any identifier and Datasets is not
            supported. If 'all', then all datasets in the Phenonaut object are merged.
            By default 'all'
        return_merged: bool
            If True the merged dataset is returned by this function. If False, then the
            new merged dataset is added to the current Phenonaut object.  By default
            False
        remove_merged : bool
            If True and return_merged is False causing the new object to be merged into
            the Phenonaut object, then source datasets which are in the current object
            (addressed by index or int in the datasets list) are removed from the
            Phenonaut object. By default True
        """

        def _check_features_match(ds_list):
            first_set = ds_list[0].features
            if all([d.features == first_set for d in ds_list[1:]]):
                return True
            return False

        if isinstance(datasets, str):
            if datasets == "all":
                datasets = list(range(len(self.datasets)))
            else:
                raise ValueError(
                    f"'{datasets}' is not a supported value for datasets, try 'all' to merge all, or include datasets, dataset names, or dataset"
                )

        if len(datasets) == 1:
            raise ValueError(
                "Cannot merge when there is only 1 dataset included in the merge"
            )

        # Check no mix of Datasets and dataset identifiers (int/str)
        if any([isinstance(d, Dataset) for d in datasets]):
            if not all([isinstance(d, Dataset) for d in datasets]):
                raise ValueError(
                    f"Cannot mix datasets with identifiers, datasets had the types: {[type(d) for d in datasets]}"
                )
            # All datasets at this point
            if not _check_features_match(datasets):
                raise ValueError(
                    "Datasets did not have the same features, could not ungroup"
                )
            new_ds = datasets[0].copy()
            new_ds.df = pd.concat([d.df for d in datasets])
            new_ds.features = (
                new_ds.features,
                f"Performed merge_datasets, {return_merged=}, {remove_merged=}",
            )
            new_ds.name = new_dataset_name
            if return_merged:
                return new_ds
            else:
                self.datasets.append(new_ds)
        else:
            # All identifiers

            if not _check_features_match([self[d] for d in datasets]):
                raise ValueError(
                    "Datasets did not have the same features, could not ungroup"
                )
            new_ds = self[datasets[0]].copy()
            new_ds.df = pd.concat([self[d].df for d in datasets])
            new_ds.features = (
                new_ds.features,
                f"Performed merge_datasets, {return_merged=}, {remove_merged=}",
            )
            new_ds.name = new_dataset_name
            if return_merged:
                return new_ds
            else:
                if remove_merged:
                    for d in reversed(datasets):
                        if isinstance(d, int):
                            del self[d]
                        else:
                            del self[self.get_dataset_index_from_name(d)]
                self.datasets.append(new_ds)

    def __repr__(self):
        return f"Phenonaut object (name = {self.name}), {len(self.datasets)} datasets, dataset names = {[d.name for d in self.datasets]})"
