# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import re
from collections import namedtuple
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from inspect import signature
from itertools import product as itertools_product
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.errors import DataError
from scipy.spatial.distance import cdist

from phenonaut import data
from phenonaut.utils import load_dict

TransformationHistory = namedtuple("TransformationHistory", ["features", "description"])


class Dataset:

    """Dataset constructor

    Dataset holds source agnostic datasets, read in using
    hints from a user configurable YAML file describing the input CSV file
    format and indicating key columns.

    Parameters
    ----------
    dataset_name : str
        Dataset name
    input_file_path : Union[Path, str, pd.DataFrame]
        Location of the input CSV/TSV/H5 file to be read, or a pd.DataFrame to use.
        If None, then an empty DataFrame object is returned. As well as CSV/TSV files,
        the location of an h5 file may also be given which contains a pandas dataframe.
        If h5 is given, then it is expected that a h5_key argument be passed.
    metadata : Union[dict, Path, str]
        Dictionary or path to yml file describing CSV file format and key
        columns. a 'sep' key:value pair may be supplied, but if absent, then the file
        is examined and if a TAB character is found present in the first line of the file,
        then it is assumed that the TAB character should be used to delimit values. This check
        is not performed if a 'sep' key is found in metadata, allowing a simple way to
        override this check. By default {}.
    kind : Optional[str]
        Instead of providing metadata, some presets are available, which make reading
        in things like DRUG-Seq easier, without the need to explicitly set all
        required transforms. If used as well as metadata, then the preset metadata
        dictionary from the kind argument is first loaded, then updated with anything in
        the metadata dictionary, this therefore allows overriding specific presets present
        in kind dictionaries. Available 'kind' dictionaries may be listed by examining:
        phenonaut.data.recipes.recipes.keys()
    init_hash : Optional[Union[str, bytes]]
        Cryptographic hashing within Phenonaut Datasets can be initialised
        with a starting/seed hash. This is useful in the creation of
        blockchain-like chains of hashes. In environments where timestamping
        is unavailable, hashes may be published and then used as input to
        subsequent experiments. Building up a provable chain along the way.
        By default None, implying an empty bytes array.
    h5_key : Optional[str]
        If input_file_path is an h5 file, then a key to access the target
        DataFrame must be supplied.
    features : Optional[list[str]]
        Features may be supplied here which are then added to the metadata dict
        if supplied.

    Raises
    ------
    FileNotFoundError
        Input CSV file not found
    DataError
        Metadata could not be used to parse input CSV
    """

    def __init__(
        self,
        dataset_name: str,
        input_file_path_or_df: Optional[Union[Path, str, pd.DataFrame]] = None,
        metadata: Union[dict, Path, str] = {},
        kind: Optional[str] = None,
        init_hash: Optional[Union[str, bytes]] = None,
        h5_key: Optional[str] = None,
        features: Optional[list[str]] = None,
    ):
        self.name = dataset_name
        self._history = []
        self._metadata = {}
        self._features = []
        self.df = None
        self.callable_functions = list(
            x for x in dir(Dataset) if callable(getattr(Dataset, x)) and x[0] != "_"
        )
        metadata = deepcopy(metadata)
        features_can_be_an_empty_list = False

        if features is not None:
            if metadata is None:
                metadata = {}

            if "features" in metadata:
                raise ValueError(
                    f"'features' argument was not None in construction of new Dataset (named {dataset_name}), and metadata contained a features entry, please use only one"
                )
            metadata["features"] = features.copy()

        if metadata is not None:
            metadata_features=metadata.get("features", None)
            if metadata_features is not None:
                if not isinstance(metadata_features, list):
                    raise ValueError(f"metadata features should be of type list, it was of type: {type(metadata_features)}")
                if metadata_features == []:
                    features_can_be_an_empty_list = True

        if init_hash is None:
            self.sha256 = sha256()
        else:
            if isinstance(init_hash, bytes):
                self.sha256 = sha256(init_hash)
            elif isinstance(init_hash, str):
                self.sha256 = sha256(init_hash.encode("utf-8"))
            else:
                raise ValueError(
                    f"The given argument init_hash was not a bytes array or a string"
                )
        # If file path is a string, make it a Path
        if isinstance(input_file_path_or_df, str):
            input_file_path_or_df = Path(input_file_path_or_df)

        # Allow creation of a blank Dataset object when None is supplied for csv_file_path
        if input_file_path_or_df is None:
            self.__update_hash()
            return

        # Load metadata.
        # If str, change to path, check exists, load yaml file defining metadata
        # Can also be a dictionary, which is allowed, or None, relying on defaults.
        if isinstance(metadata, (Path, str)):
            metadata = load_dict(metadata)
        if metadata is None:
            metadata = {}

        if kind is not None:
            from .recipes import kinds

            regularised_kind = (
                kind.lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("_", "")
                .replace(".", "")
            )
            if regularised_kind not in kinds:
                raise ValueError(
                    f"Dataset kind given in kind argument was '{kind}',"
                    f" but this was not found in phenonaut. Usable kind arguments are: {kinds.keys()}"
                )
            metadata = {**kinds[regularised_kind], **metadata}

        # skip_row_numbers and header_row_number in metadata must be a list
        if isinstance(metadata.get("skip_row_numbers"), int):
            metadata["skip_row_numbers"] = [metadata["skip_row_numbers"]]
        if isinstance(metadata.get("header_row_number"), int):
            metadata["header_row_number"] = [metadata["header_row_number"]]

        if isinstance(input_file_path_or_df, Path):
            if input_file_path_or_df.suffix == ".h5":
                if "key" not in metadata:
                    raise KeyError(
                        "Attempting to load hdf5 file, but no key was given in metadata"
                    )
                self.df = pd.read_hdf(input_file_path_or_df, h5_key)
            else:
                # Read the CSV file into a pandas DataFrame. It needs to be read in
                # before features are set because the user may be using
                # features_prefix, so we need to examine column names to obtain
                # feature names

                # Pass everything in metadata that can be passed to pd.read_csv
                read_csv_args = {
                    ar: ar_val
                    for ar, ar_val in metadata.items()
                    if ar in signature(pd.read_csv).parameters.keys()
                }

                if "sep" not in read_csv_args:
                    # See if we can find a tab in the first line:
                    with open(input_file_path_or_df) as myfile:
                        head = next(myfile)
                        if "\t" in head:
                            read_csv_args["sep"] = "\t"

                self.df = pd.read_csv(input_file_path_or_df, **read_csv_args)

            # Some datasets contain column names with spaces after them. Additionally,
            # Columbus column titles often have \xa0 at the end.  Strip this whitespace.
            self.df = self.df.rename(columns=lambda x: x.strip())

        if isinstance(input_file_path_or_df, pd.DataFrame):
            self.df = input_file_path_or_df.copy()

        # Some documentation and early users rely upon transpose being in metadata.
        # If it is, then we move it to transforms within metadata.
        if "transpose" in metadata:
            if "transforms" not in metadata:
                metadata["transforms"] = []
            metadata["transforms"].extend([("transpose", True)])

        # Perform all transformations
        for t in metadata.get("transforms", []):
            if isinstance(t, str):
                # If string, like "transpose", then make it into a tuple
                t = (t,)

            if isinstance(t, dict):
                if len(t.keys()) != 1:
                    raise ValueError(
                        "Transforms passed as dicts should contain one item, "
                        "where the key is the function name to call. This is "
                        "to stop multiple calls to the same function "
                        "overwriting each other due to the behaviour of "
                        "dictionary keys"
                    )
                t = next(iter(t.items()))

            # At this stage, it should be a list or tuple, or at least
            # something compatible with __getitem__ and with a __len__ method
            # Avoiding using isinstance as much as possible, relying on duck
            # typing.
            # Could be of the form:
            #  ('replace_str',('Var2', 'Pos_', 'Pos-')),
            #  ('replace_str', 3, 'Pos-', 'Pos_'),
            #  ('transpose',)
            if len(t) == 0:
                continue
            elif len(t) == 1:
                func_name, func_args = t[0], ()
            elif len(t) == 2:
                if not isinstance(t[1], (list, tuple)):
                    func_name, func_args = t[0], (t[1],)
                else:
                    func_name, func_args = t
            if len(t) > 2:
                func_name, func_args = t[0], t[1:]

            if func_name not in self.callable_functions:
                raise KeyError(
                    f"'transforms' contains an unknown function: '{func_name}'"
                )

            if isinstance(func_args, dict):
                getattr(self, func_name)(**func_args)
            else:
                getattr(self, func_name)(*func_args)

        if "features" in metadata and "initial_features" in metadata:
            raise KeyError(
                "Both 'features' and 'initial_features' found in provided metadata. Please provide only one, if present, 'features' is renamed 'initial_features'"
            )

        # Move features to initial_features
        if "features" in metadata:
            if not isinstance(metadata["features"], (list, tuple, np.ndarray)):
                raise ValueError(
                    f"'features' must be a list, np.nparray, or tuple of strings, it was {type(metadata['features'])}"
                )
            metadata["initial_features"] = metadata["features"].copy()
            del metadata["features"]

        # If no features were given, use features_prefix and features_regex to identify them
        if "initial_features" not in metadata:
            features_prefix = metadata.get("features_prefix")
            features_regex = metadata.get("features_regex")

            if features_prefix is None and features_regex is None:
                print(
                    "No 'features', 'features_prefix', or 'features_regex' given, assuming prefix is 'feat_'"
                )
                features_prefix = ["feat_"]
            if isinstance(features_prefix, str):
                features_prefix = [features_prefix]

            metadata["initial_features"] = []
            if features_prefix is not None:
                # Extract features using features_prefix
                metadata["initial_features"].extend(
                    [
                        column_name
                        for column_name in self.df.columns.to_list()
                        if str(column_name).startswith(tuple(features_prefix))
                    ]
                )

            if features_regex is not None:
                metadata["initial_features"].extend(
                    f
                    for f in list(
                        filter(re.compile(features_regex).match, self.df.columns.values)
                    )
                    if f not in metadata["initial_features"]
                )

            if len(metadata["initial_features"]) == 0:
                message = f"No column headers found with specified prefix(es): {features_prefix}, or matching features_regex {features_regex}, first 10 column headers were: {self.df.columns[:10]}"
                raise DataError(message)

        if metadata["initial_features"] is None:
            raise DataError("features cannot be None")
        if len(metadata["initial_features"]) == 0:
            if features_can_be_an_empty_list:
                print(
                    f"Warning, the use of no features was explicitly requested for {dataset_name}"
                )
            else:
                raise DataError(
                    f"features cannot be an empty list, features = {metadata['initial_features']}"
                )

        # Sort features using a natural sort, see:
        # https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
        # The aim is to make
        # ['f1', 'f10', 'f13', 'f2a', 'f03']
        # sort as:
        # ['f1', 'f2a', 'f03', 'f10', 'f13']
        # ignoring zero padding etc.
        def natural_sort_convert_digit_or_str(x):
            return int(x) if x.isdigit() else x.lower()

        metadata["initial_features"] = sorted(
            metadata["initial_features"],
            key=lambda x: [
                natural_sort_convert_digit_or_str(c) for c in re.split("([0-9]+)", x)
            ],
        )

        # Check all specified features are present in the DataFrame
        missing_features = []
        for feature in metadata["initial_features"]:
            if feature not in self.df.columns:
                print(f"Specified feature ({feature}) not found in columns")
                missing_features.append(feature)
        if len(missing_features) != 0:
            raise ValueError(
                f"The following features were not identified as columns: {missing_features}"
            )

        # At this point in the code, there was a check to ensure that all feature columns contained numeric types.
        # For the CMAP dataset, this took a long time >45 mins using:
        # self.df[metadata["initial_features"]] = self.df[metadata["initial_features"]].apply(
        #     pd.to_numeric, errors="coerce"
        # A supposedly quicker approach tried was:
        # self.df[metadata["initial_features"]] = self.df[metadata["initial_features"]].astype(float)
        # But this took 20 mins.  It was therefore decided to remove the check and rely on Pandas to
        # correctly assign numeric types if it can - this is correct, expected behavior anyway.

        if isinstance(input_file_path_or_df, Path):
            self.features = (
                metadata["initial_features"],
                f"{input_file_path_or_df=}, {metadata=}, {init_hash=}",
            )

        if isinstance(input_file_path_or_df, pd.DataFrame):
            self.features = (
                metadata["initial_features"],
                f"DataFrame passed to Dataset constructor, {metadata=}, {init_hash=}",
            )
        self._metadata = metadata
        self.__update_hash()

    @property
    def data(self):
        """Return self.df[self.features]

        Returns
        -------
        pd.DataFrame
            DataFrame containing only features and index
        """
        return self.df[self.features]

    @property
    def features(self):
        """Return current dataset features

        Returns
        -------
        list
            List of strings containing current features
        """
        return deepcopy(self._features)

    @features.setter
    def features(self, features_and_message_tuple: tuple):
        """Assign new set of feature columns and record a history message

        As well as recording new features and the history message, setting new
        features using this setter causes the hash of the dataset to be updated.

        Parameters
        ----------
        features_and_message_tuple : tuple(list, str)
            Tuple, with the first element being a list of strings documenting
            new features, and the second element is a history message
            documenting what was done to arrive at the new features.
        """

        if not isinstance(features_and_message_tuple, tuple):
            raise AttributeError(
                "Setting features requires a tuple to be supplied of the form:([features], 'history message')"
            )
        if len(features_and_message_tuple) != 2:
            raise AttributeError(
                "Supplied tuple did not have 2 elements. Setting features requires a tuple to be supplied of the form:([features], 'history message')"
            )
        if len(features_and_message_tuple[0]) == 0:
            print(
                f"Features list is empty, message was {features_and_message_tuple[1]}"
            )
        if not isinstance(features_and_message_tuple[1], str):
            raise AttributeError(
                "Message (second tuple element) must be of type str. Setting features requires a tuple to be supplied of the form:([features], 'history message')"
            )

        self._history.append(TransformationHistory(*features_and_message_tuple))
        self._features = list(features_and_message_tuple[0])
        self.__update_hash()

    def get_history(self) -> list[TransformationHistory]:
        """Get dataset history

        Returns
        -------
        list[TransformationHistory]
            List of TransformationHistory (named tuples) which contain a list
            of features as the first element and then a plain text description
            of what was applied to arrive at those features as the second
            element.
        """
        return self._history

    @property
    def history(self):
        """Get dataset history

        Returns the same as calling .get_history on the dataset

        Returns
        -------
        list[TransformationHistory]
            List of TransformationHistory (named tuples) which contain a list
            of features as the first element and then a plain text description
            of what was applied to arrive at those features as the second
            element.

        """
        return self.get_history()

    @property
    def perturbation_column(self):
        """Return the name of the treatment column

        A treatement is an identifier relating to the peturbation. In many
        cases, it is the unique compound name or identifier.  Many replicates
        may be present, with identifiers like 'DMSO' etc.

        Returns
        -------
        String
            Column name of dataframe containing the treatment.
        """
        perturbation_column = self._metadata.get("perturbation_column", None)
        if perturbation_column is None:
            print(
                f"Warning: Attempted to get dataset ({self.name=}) perturbation_column, but it is None"
            )
        return perturbation_column

    def __str__(self):
        return f"Phenonaut.Dataset:'{self.name}', shape={self.df.shape}, n_features={len(self.features)}"

    def __repr__(self):
        return f"Phenonaut.Dataset:'{self.name}', shape={self.df.shape}, n_features={len(self.features)}"

    @perturbation_column.setter
    def perturbation_column(self, perturbation_column: str):
        """Assign new treatment column name

        A treatement is an identifier relating to the peturbation. In many
        cases, it is the unique compound name or identifier.  Many replicates
        may be present, with identifiers like 'DMSO' etc.

        Parameters
        ----------
        perturbation_column : str
            New column name which contains the peturbation IDs.
        """
        self._metadata["perturbation_column"] = perturbation_column
        self.features = (
            self.features,
            f"Updated perturbation column, perturbation column is now {perturbation_column}",
        )

    def get_unique_perturbations(self):
        if self.perturbation_column is None:
            return None
        return [
            t
            for t in self.df[self.perturbation_column].unique()
            if not pd.isna(t) and t != ""
        ]

    def get_feature_ranges(self, pad_by_percent: Union[float, int]) -> tuple:
        """Get the ranges of feature columns

        Parameters
        ----------
        pad_by_percent : Union[float, int]
            Optionally, pad the ranges by a percent value

        Returns
        -------
        tuple
            Returns tuple of tuples with shape (features, 2), for example,
            with two features it would return:
            ((min_f1, max_f1), (min_f2, max_f2))
        """
        feature_min_max = []
        for feature in self.features:
            feature_min = self.df[feature].min()
            feature_max = self.df[feature].max()
            feature_range = feature_max - feature_min
            feature_min_max.append(
                (
                    feature_min - ((feature_range / 100) * float(pad_by_percent)),
                    feature_max + ((feature_range / 100) * float(pad_by_percent)),
                )
            )
        return tuple(feature_min_max)

    def get_non_feature_columns(self) -> list:
        """Get columns which are not features

        Returns
        -------
        list[str]
            Returns list of Dataset columns which are not currently features.
        """
        return list(set(self.df.columns.unique().values) - set(self.features))

    def drop_columns(
        self,
        column_labels: Union[str, list[str], tuple[str]],
        reason: Optional[str] = None,
    ) -> None:
        """Drop columns inplace, update features if needed and set new history.

        Intelligently drop columns from the dataset (inplace). If any of those
        columns were listed as features, then remove them from the features list
        and set a new new history.  Updating features and new history only
        happens if it needs to (removed column was a feature). Updaing of
        features will cause hash update.

        Parameters
        ----------
        column_labels : Union[str, list[str], tuple[str]]
            List of column labels which should be removed. Can also be a str to
            remove just one column.
        reason : Optional[str]
            A reason may be given for dropping the column. If not None and the
            column was a feature, then this reason is recorded along with the
            history. If None, and the column was a feature, then the history
            entry will state:
            "Droped columns ({column_labels})"
            Where column_labels contains the dropped columns.
            If reason is not None and the column is a feature, then the history
            entry will state:
            "Droped columns ({column_labels}), reason:{reason}"
            where {reason} is the given reason. Has no effect if the dropped
            column is not a feature, or the list of dropped columns do not
            contain a feature. By default None.
        """

        if isinstance(column_labels, str):
            column_labels = [column_labels]
        features_removed = set(self.features).intersection(set(column_labels))

        self.df.drop(columns=column_labels, inplace=True)

        reason_message = (
            f", reason = {reason}"
            if reason is not None and len(features_removed) > 0
            else ""
        )

        if len(features_removed) > 0:
            new_features = [
                feat for feat in self.features if feat not in features_removed
            ]
            self.features = (
                new_features,
                f"Dropped columns ({column_labels}) {reason_message}",
            )

    def drop_rows(self, row_indices: pd.Index) -> None:
        """Drop rows inplace given a set of indices.

        Intelligently drop rows from the dataset (inplace). Updating of
        rows will not cause hash update as features are unchanged.

        Parameters
        ----------
        row_indices : pd.Index
            List of row indexes which should be removed. Can also be an int
            to remove just one row.

        Raises
        ------
            KeyError: Error raised if the index is missing from dataframe index
        """

        missing_indices = set(row_indices) - set(self.df.index)
        if missing_indices:
            raise KeyError(
                f"Could not find {missing_indices} in dataframe indices, which are: {self.df.index.values}"
            )

        self.df.drop(index=row_indices, inplace=True)

    def filter_inplace(self, query: str) -> None:
        """Apply a pandas query style filter, keeping all that pass

        Parameters
        ----------
        query : str
            Pandas style query which when applied to rows, will keep all those
            which return True.
        """
        self.df = self.df.loc[self.df.query(query).index]
        self.features = self.features, f"Filtered in place, query={query}"

    def subtract_func_results_on_features(
        self,
        query_or_perturbation_name: str,
        groupby: Optional[Union[str, list[str]]],
        func: Union[Callable, str, None] = "median",
    ) -> None:
        """Subtract the result of a function applied to rows

        Useful function for centering plates on DMSO or control perturbations.
        If called with no func, then median is taken as the required function.
        The median, or result of applied function to rows identified by the query
        string (query_or_perturbation_name parameter) are subtracted from all
        perturbations. The query_or_perturbation_name may also be an
        identifier present in the datasets perturbation column (if set). If a
        column name, or list of column names are given in the groupby argument,
        then the operation is carried out within these groups before being
        merged back to the original dataframe.

        Parameters
        ----------
        query_or_perturbation_name : str
            Pandas style query to retrieve rows from which quantities for
            substraction are calculated, or, if the dataset has
            perturbation_column set and the parameter value can be found it the
            perturbation column, then these samples are used and have the given
            function applied to them.  In short, for a Dataset with
            perturbation_column set to "cpd_name", then the same effect can be
            achied with this parameter being "DMSO" and "cpd_name=='DMSO'".
        groupby : Optional[str, list[str]]
            The name, or list of names of columns that the DataSet should be
            grouped by for application of the transformation on a group-by group
            basis. This is very useful if neededing to subtract median DMSO
            perturbation features on a plate-by-plate basis, whereby the column
            containing plateIDs would be supplied.  Multiple column names may
            also be supplied.
        func: Union[Callable, str, None]
            The callable to use in calculation of the quantity to subtract for
            each perturbation. Special cases exist for 'median' and 'mean' strings
            whereby pd.median and pd.mean are applied respectively.
            If None, then no action is taken. By default 'median'.
        """

        def _perform_df_transform(all_df, q, feat, func) -> pd.DataFrame:
            df_query = all_df.query(q)
            if df_query.shape[0] == 0:
                raise DataError(
                    "Query or perturbation returned no matches when attempting to find amounts of features for subtraction in the rest of group.  If using groupby - is the requested perturbation present in the group?"
                )
            if isinstance(func, str):
                if func == "mean":
                    all_df.loc[:, feat] = all_df[feat] - df_query[feat].mean()
                    return all_df
                elif func == "median":
                    all_df.loc[:, feat] = all_df[feat] - df_query[feat].median()
                    return all_df
                else:
                    raise ValueError(
                        f"Unknown magic string passed ('{func}'), options are mean or median"
                    )
            else:
                all_df.loc[:, feat] = all_df[feat] - df_query[feat].apply(func)
            return all_df

        # In the line below, get pertibation column this way rather than use the getter to avoid warning message if None.
        if (
            self._metadata.get("perturbation_column", None) is not None
            and query_or_perturbation_name
            in self.df[self._metadata.get("perturbation_column", None)]
        ):
            query_or_perturbation_name = (
                f"{self.perturbation_column}=='{query_or_perturbation_name}'"
            )
        if groupby is None:
            self.df[self.features] = _perform_df_transform(
                self.df, self.df.query(query_or_perturbation_name), self.features, func
            )
        else:
            grouped_df_results = []
            for _, g in self.df.groupby(groupby):
                grouped_df_results.append(
                    _perform_df_transform(
                        g, query_or_perturbation_name, self.features, func
                    )
                )
            self.df = pd.concat(grouped_df_results)

        self.features = (
            self.features,
            f"Subtracted features of rows matching perturbation: {query_or_perturbation_name}, func={func}",
        )

    def subtract_median(
        self, query_or_perturbation_name: str, groupby: Optional[Union[str, list[str]]]
    ) -> None:
        """Subtract the median of rows identified in the query from features

        Useful function for centering plates on DMSO or control perturbations.
        The median of row features identified by the query string
        (query_or_perturbation_name parameter) are subtracted from all
        perturbations. If the query_or_perturbation_name may also be an
        identifier present in the datasets perturbation column (if set). If a
        column name, or list of column names aregiven in the groupby argument,
        then the operation is carried out within these groups before being
        merged back to the original dataframe.

        Parameters
        ----------
        query_or_perturbation_name : str
            Pandas style query to retrieve rows from which quantities for
            substraction are calculated, or, if the dataset has
            perturbation_column set and the parameter value can be found it the
            perturbation column, then these samples are used and have the given
            function applied to them.  In short, for a Dataset with
            perturbation_column set to "cpd_name", then the same effect can be
            achied with this parameter being "DMSO" and "cpd_name=='DMSO'".
        groupby : Optional[str, list[str]]
            The name, or list of names of columns that the DataSet should be
            grouped by for application of the transformation on a group-by group
            basis. This is very useful if neededing to subtract median DMSO
            perturbation features on a plate-by-plate basis, whereby the column
            containing plateIDs would be supplied.  Multiple column names may
            also be supplied.
        """
        self.subtract_func_results_on_features(
            query_or_perturbation_name=query_or_perturbation_name,
            groupby=groupby,
            func="median",
        )

    def subtract_mean(
        self, query_or_perturbation_name: str, groupby: Optional[Union[str, list[str]]]
    ) -> None:
        """Subtract the mean of rows identified in the query from features

        Useful function for centering plates on DMSO or control perturbations.
        The mean of row features identified by the query string
        (query_or_perturbation_name parameter) are subtracted from all
        perturbations. If the query_or_perturbation_name may also be an
        identifier present in the datasets perturbation column (if set). If a
        column name, or list of column names aregiven in the groupby argument,
        then the operation is carried out within these groups before being
        merged back to the original dataframe.

        Parameters
        ----------
        query_or_perturbation_name : str
            Pandas style query to retrieve rows from which quantities for
            substraction are calculated, or, if the dataset has
            perturbation_column set and the parameter value can be found it the
            perturbation column, then these samples are used and have the given
            function applied to them.  In short, for a Dataset with
            perturbation_column set to "cpd_name", then the same effect can be
            achied with this parameter being "DMSO" and "cpd_name=='DMSO'".
        groupby : Optional[str, list[str]]
            The name, or list of names of columns that the DataSet should be
            grouped by for application of the transformation on a group-by group
            basis. This is very useful if neededing to subtract mean DMSO
            perturbation features on a plate-by-plate basis, whereby the column
            containing plateIDs would be supplied.  Multiple column names may
            also be supplied.
        """
        self.subtract_func_results_on_features(
            query_or_perturbation_name=query_or_perturbation_name,
            groupby=groupby,
            func="mean",
        )

    def divide_median(self, query: str) -> None:
        """Divide dataset features by the median of rows identified in the query

        Useful function for normalising to controls.

        Parameters
        ----------
        query : str
            Pandas style query to retrieve rows from which medians are
            calculated.
        """
        self.df[self.features] = (
            self.data / self.df.query(query)[self.features].median()
        )
        self.features = (
            self.features,
            f"Divided by median of rows matching query: {query}",
        )

    def divide_mean(self, query: str) -> None:
        """Divide dataset features by the mean of rows identified in the query

        Useful function for normalising to controls.

        Parameters
        ----------
        query : str
            Pandas style query to retrieve rows from which means are calculated.
        """
        self.df[self.features] = self.data / self.df.query(query)[self.features].mean()
        self.features = (
            self.features,
            f"Divided by mean of rows matching query: {query}",
        )

    def distance_df(
        self,
        candidate_dataset: "Dataset",
        metric: Union[str, Callable] = "euclidean",
        return_best_n_indexes_and_score: Optional[int] = None,
        lower_is_better=True,
    ) -> pd.DataFrame:
        """Generate a distance DataFrame

        Distance DataFrames allow simple generation of pd.DataFrames where the
        index take the form of perturbations and the columns other
        perturbations. The values at the intersections are therfore the
        distances between these perturbations in feature space. Many different
        metrics both inbuilt and custom/user defined may be used.

        Parameters
        ----------
        candidate_dataset : Dataset
            The dataset to which the query (this) should be compared.
        metric : Union[str, Callable], optional
            Metric which should be used for the distance calculation.  May be a
            simple string understood by scipy.spatial.distance.cdist, or a
            callable, like a function or lambda accepting two vectors
            representing query and candidate features. By default "euclidean".
        return_best_n_indexes_and_score : Optional[int], optional
            If an integer is given, then just that number of best pairs/measures
            are returned. By default None.
        lower_is_better : bool, optional
            If using the above 'return_best_n_indexes_and_score' then it needs
            to be flagged if lower is better (default), or higher is better.
            By default True

        Returns
        -------
        Union [pd.DataFrame, tuple(tuple(int, int), float)]
            Returns a distance Dataframe, unless
            'return_best_n_indexes_and_score' is an int, in which case a list of
            the top scoring pairs are returned in the form of a nested tuple:
            ((from, to), score)

        Raises
        ------
        ValueError
            Error raised if this Dataset and the given candidate Dataset do not
            share common features.
        """
        if not all([f in candidate_dataset.features for f in self.features]):
            raise ValueError("Datasets do not share common features")

        dist_mat = cdist(self.data, candidate_dataset.df[self.features], metric=metric)

        if return_best_n_indexes_and_score is None:
            return pd.DataFrame(
                data=dist_mat,
                index=self.df.index,
                columns=candidate_dataset.df.index.to_list(),
            )

        else:
            if lower_is_better:
                top_n = list(
                    map(
                        lambda x: np.unravel_index(x, dist_mat.shape),
                        np.argsort(dist_mat, axis=None)[
                            :return_best_n_indexes_and_score
                        ],
                    )
                )
            else:
                top_n = list(
                    map(
                        lambda x: np.unravel_index(x, dist_mat.shape),
                        np.argsort(dist_mat, axis=None)[
                            -return_best_n_indexes_and_score:
                        ],
                    )
                )[::-1]
        return [(loc, dist_mat[loc[0], loc[1]]) for loc in top_n]

    def get_ds_from_query(self, name: str, query: str):
        """Make a new Dataset object from a pandas style query.

        Parameters
        ----------
        name : str
            Name of new dataset
        query : str
            Pandas style query from which all rows returning true will be
            included into the new PhenonautGenericData set object.

        Returns
        -------
        Dataset
            New dataset created from query
        """
        new_ds = Dataset(name, input_file_path_or_df=None, metadata=None)
        new_ds.df = self.df.query(query)
        new_ds.features = (self.features, f"New dataset from query: {query}")
        new_ds._metadata = self._metadata
        return new_ds

    def copy(self):
        """Return a deep copy of the Dataset object

        Returns
        -------
        PhenonautData
            Copy of the input object.
        """
        tmp_hash_object = self.sha256.copy()
        self.sha256 = None
        new_ds = deepcopy(self)
        self.sha256 = tmp_hash_object
        new_ds.sha256 = self.sha256.copy()
        return new_ds

    def filter_columns_with_prefix(
        self, column_prefix: Union[str, list], keep: bool = False
    ):
        """Filter columns based on prefix

        Parameters
        ----------
        column_prefix : Union[str, list]
            Prefix for columns as a string, or alternatively, a list of string
            prefixes
        keep : bool, optional
            If true, only columns matching the prefix are kept, if false, these
            columns are removed, by default False
        """
        column_names = []
        if isinstance(column_prefix, list):
            column_prefix = tuple(column_prefix)
        column_names = [
            c for c in self.df.columns.values if c.startswith(column_prefix)
        ]
        self.filter_columns(column_names, keep=keep)

    def filter_columns(self, column_names: list, keep=True, regex=False):
        """Filter dataframe columns

        Parameters
        ----------
        column_names : list
            Column names
        keep : bool, optional
            Keep columns listed in column_names, if false, then the opposite
            happens and these columns are removed, by default True.
        """

        if regex:
            new_column_names = []
            for pattern in column_names:
                for match in list(
                    filter(re.compile(pattern).match, self.df.columns.values)
                ):
                    if match not in new_column_names:
                        new_column_names.append(match)
            column_names = new_column_names

        if len(column_names) == 0:
            return
        if keep:
            columns_for_removal = set(self.df.columns.values) - set(column_names)
            features_to_remove = set(self.features).intersection(columns_for_removal)
            if len(features_to_remove) > 0:
                self.features = (
                    [f for f in self.features if f not in features_to_remove],
                    f"Features removed due to filtering of columns, columns kept were {column_names=}, resulting in removal of the following columns  {columns_for_removal=}, which removed the following features {features_to_remove}",
                )
            self.df = self.df[column_names]

        else:
            features_to_remove = set(self.features).intersection(column_names)
            if len(features_to_remove) > 0:
                self.features = (
                    [f for f in self.features if f not in features_to_remove],
                    f"Features removed due to filtering of columns, columns removed were {column_names=}, which removed the following features {features_to_remove}",
                )
            self.df = self.df.drop(column_names, axis=1)

    def filter_rows(
        self, query_column: str, values: Union[list, str], keep: bool = True
    ):
        """Filter dataframe rows

        Parameters
        ----------
        query_column : str
            Column name which is being filtered on
        values : Union[list, str]
            List or string of values to be filtered on
        keep : bool, optional
            If true, then only rows containing listed values in query column
            are kept. If this argument is false, then the opposite occurs, and
            the rows matching are discarded, by default True

        """

        if not isinstance(values, list):
            values = [values]

        if not query_column in self.df.columns:
            raise KeyError(
                f"Could not find {query_column} in dataframe columns, columns are: {self.df.columns.values}"
            )
        if keep:
            self.df = self.df.query(f"{query_column} in @values")
        else:
            self.df = self.df.query(f"{query_column} not in @values")
        self.features = (self.features, f"Filtered rows, {query_column=}, {values=}")

    def rename_column(self, from_column_name: str, to_column_name: str):
        """Rename a single dataset column

        Parameters
        ----------
        from_column_name : str
            Name of column to rename
        to_column_name : str
            New column name
        """
        self.rename_columns({from_column_name: to_column_name})

    def rename_columns(self, from_to: dict):
        """Rename multiple columns

        Parameters
        ----------
        from_to : dict
            Dictionary of the form {'old_name':'new_name'}
        """
        self.df.rename(columns=from_to, inplace=True)
        if len(self._history) != 0:
            self.features = self.features, f"Renamed columns: {from_to}"

        # Renamed column might be a feature, if so, then rename the feature
        from_to_in_features = {
            k_f: from_to[k_f] for k_f in from_to if k_f in self.features
        }
        if len(from_to_in_features) > 0:
            self.features = (
                [from_to_in_features.get(f, f) for f in self.features],
                "New features set by renaming columns ({from_to_dict})",
            )

    def df_to_csv(self, output_path: Union[Path, str], **kwargs):
        """Write DataFrame to CSV

        Convenience function to write the underlying DataFrame to a CSV.  Additional
        arguments will be passed to the Pandas.DataFrame.to_csv function.

        Parameters
        ----------
        output_path : Union[Path, str]
            Target output file
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)
        if "index" not in kwargs:
            kwargs["index"] = False
        self.df.to_csv(output_path, **kwargs)

    def df_to_multiple_csvs(
        self,
        split_by_column: str,
        output_dir: Optional[Union[str, Path]] = None,
        file_prefix: str = "",
        file_suffix="",
        file_extension=".csv",
        **kwargs,
    ):
        """Wite multiple CSV files from a dataset DataFrame.

        In the case where one output CSV is required per plate, then splitting
        the underlying DataFrame on something like a PlateID serves the purpose
        of generating one output CSV file per plate.  This can be achieved with
        this function and providing the column to split on.

        Parameters
        ----------
        split_by_column : str
            Column containing unique values within a split output CSV file
        output_dir : Optional[Union[str, Path]], optional
            Target output directory for split CSV files, by default None
        file_prefix : str, optional
            Prefix for split CSV files, by default ""
        file_suffix : str, optional
            Suffix for split CSV files, by default ""
        file_extension : str, optional
            File extension for split CSV files, by default ".csv"
        """

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

        if "index" not in kwargs:
            kwargs["index"] = False

            splits = self.df[split_by_column].unique()
            for split in splits:
                if split == np.nan:
                    continue
                split_output_path = (
                    output_dir / f"{file_prefix}{split}{file_suffix}{file_extension}"
                )
                self.df.query(f"{split_by_column} == '{split}'").to_csv(
                    split_output_path, **kwargs
                )

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
        """Add standard well IDs - such as A1, A2, etc.

        If a dataset contains numerical row and column names, then they may be
        translated into standard letter-number well IDs.

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
            Do not resort the dataset by well ID, by default False.
        """
        if plate_type not in [384]:
            message = f"Attempted to assign alphanumeric wells (like B01) to requested {plate_type} well plate.  {plate_type} well plates not implemented."
            raise ValueError(message)

        MAX_ROWS = None
        MAX_COLUMNS = None
        row_map = None
        column_map = None
        if plate_type == 384:
            MAX_ROWS = 16
            MAX_COLUMNS = 24
            row_map = {n: chr(n + 64) for n in range(1, MAX_ROWS + 1)}
            column_map = {n: f"{n}".zfill(2) for n in range(1, MAX_COLUMNS + 1)}

        # Make sure to cast to a list, otherwise it is consumed upon use/iteration
        all_possible_well_ids = list(
            itertools_product(row_map.keys(), column_map.keys())
        )
        numerical_column = self.df[numerical_column_name].astype(int)
        numerical_row = self.df[numerical_row_name].astype(int)
        if numerical_column.max() > MAX_COLUMNS:
            raise DataError(
                f"Column value {numerical_column.max()} is greater than maximum for {plate_type} well plate of {MAX_COLUMNS}"
            )
        if numerical_row.max() > MAX_ROWS:
            raise DataError(
                f"Row value {numerical_row.max()} is greater than maximum for {plate_type} well plate of {MAX_ROWS}"
            )

        self.df[new_well_column_name] = numerical_row.apply(
            lambda x: row_map[x]
        ).astype(str) + numerical_column.apply(lambda x: column_map[x]).astype(str)

        if add_empty_wells:
            if plate_barcode_column is None:
                all_well_ids = list(zip(numerical_row, numerical_column))
                missing_wells = list(set(all_possible_well_ids) - set(all_well_ids))
                if len(missing_wells) > 0:
                    additional_data = pd.DataFrame.from_dict(
                        {
                            numerical_row_name: [w[0] for w in missing_wells],
                            numerical_column_name: [w[1] for w in missing_wells],
                            new_well_column_name: [
                                row_map[w[0]] + column_map[w[1]] for w in missing_wells
                            ],
                        }
                    )
                    self.df = pd.concat(
                        [self.df, additional_data], axis=0, ignore_index=True
                    )
                    if not no_sort:
                        self.df = self.df.sort_values([new_well_column_name])
            else:
                for plate in self.df[plate_barcode_column].unique():
                    if plate is np.nan:
                        continue
                    plate_df = self.df.query(f"{plate_barcode_column} == '{plate}'")
                    all_well_ids = list(
                        zip(
                            plate_df[numerical_row_name],
                            plate_df[numerical_column_name],
                        )
                    )
                    missing_wells = [
                        (w[0], w[1], plate)
                        for w in (set(all_possible_well_ids) - set(all_well_ids))
                    ]

                    if len(missing_wells) > 0:
                        additional_wells = pd.DataFrame.from_dict(
                            {
                                numerical_row_name: [w[0] for w in missing_wells],
                                numerical_column_name: [w[1] for w in missing_wells],
                                new_well_column_name: [
                                    row_map[w[0]] + column_map[w[1]]
                                    for w in missing_wells
                                ],
                                plate_barcode_column: [w[2] for w in missing_wells],
                            }
                        )
                        self.df = pd.concat(
                            [self.df, additional_wells], axis=0, ignore_index=True
                        )
                        if not no_sort:
                            self.df = self.df.sort_values(
                                [plate_barcode_column, new_well_column_name]
                            )
        self.features = (
            self.features,
            f"Added well IDs, {numerical_column_name=}, {numerical_row_name=}, {plate_type=}, {new_well_column_name}, {add_empty_wells=}, {plate_barcode_column=}, {no_sort=}",
        )

    def subtract_median_perturbation(
        self,
        perturbation_label: str,
        per_column_name: Optional[str] = None,
        new_features_prefix: str = "SMP_",
    ):
        r"""Subtract the median perturbation from all features

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
        new_features = [f"SMP_{feat}" for feat in self.features]
        if per_column_name is None:
            # TODO: this should throw an error/warning because of dimension mismatch
            self.df[new_features] = self.data - self.df.median(axis=0)
        else:
            for unique_plate in self.df[per_column_name].unique():
                medians = self.df.loc[
                    self.df[per_column_name] == unique_plate, self.features
                ].median(axis=0)
                self.df.loc[
                    self.df[self.df[per_column_name] == unique_plate].index,
                    new_features,
                ] = (
                    self.df.loc[
                        self.df[self.df[per_column_name] == unique_plate].index,
                        self.features,
                    ]
                    - medians
                ).values
        self.features = (
            new_features,
            f"Subtracted median perturbation {perturbation_label=}, {per_column_name=}",
        )

    def transpose(self, reset_index: bool = True, new_header_column: Optional[int] = 0):
        """Transpose internal DataFrame"""

        self.df = self.df.transpose()
        if reset_index:
            self.df = self.df.reset_index()

        if new_header_column is not None:
            self.df.columns = self.df.iloc[new_header_column]
            self.df.drop(index=new_header_column, inplace=True)

    def replace_str(self, column: Union[str, int], pat: str, repl: str):
        """Replace a string present in a column

        Parameters
        ----------
        column : Union[str, int]
            Name of the column(could be a feature), within which to search and replace instances of the string specified in the 'pat' argument.
        pat : str
            The patter, (non-regex), just query substring to find and replace.
        repl : str
            Replacement text for the substring identified in the 'pat' argument.
        """
        if isinstance(column, int):
            print(f"replacing in {column}, {self.df.columns.tolist()}")
            column = self.df.columns.tolist()[column]
        self.df[column] = self.df[column].str.replace(pat, repl)

    def split_column(self, column: Union[str, int], pat: str, new_columns: list[str]):
        """Split a column on a delimiter

        If a column named 'data' contained:

        === ===============
        idx data
        === ===============
        1   A2_CPD1_Plate1
        === ===============

        Then calling:

        .. code-block:: python

            split_column('data', '_', ['WellID', 'CpdID', 'PlateID'])

        Would introduce the following new columns into the dataframe:

        === ====== ===== =======
        idx WellID CpdID PlateID
        === ====== ===== =======
        1   A2     CPD1  Plate1
        === ====== ===== =======

        Parameters
        ----------
        column : Union[str, int]
            Name of column to split, or the index of the column.
        pat : str
            Pattern (non-regex), usually a delimiter to split on.
        new_columns : list[str]
            List of new column names. Should be the correct size to absorb all
            produced splits.

        Raises
        ------
        DataError
            Inconsistent number of splits produced when splitting the column.
        ValueError
            Incorrect number of new column names given in new_columns.
        """
        if isinstance(column, int):
            column = self.df.columns.tolist()[column]
        pat_count = self.df[column].str.count("_").unique().tolist()
        if len(pat_count) != 1:
            different_count_indexes = [
                self.df.loc[
                    self.df.loc[self.df[column].str.count(pat) == c].index[0], column
                ]
                for c in pat_count
            ]

            raise DataError(
                f"Attempting to split {column} column on '{pat}', "
                f"but column does not contain a consistent count of '{pat}',"
                f" contained: {pat_count} instances, from {column} values such as:\n"
                f"{different_count_indexes}"
            )
        if len(new_columns) != pat_count[0] + 1:
            raise ValueError(
                f"'new_columns' argument to split_column contains "
                f"{len(new_columns)} elements, but splitting on '{pat}' "
                f"produces {pat_count[0]} new columns"
            )

        self.df[new_columns] = self.df[column].str.split(pat, expand=True)

    def pivot(self, feature_names_column: str, values_column: str):
        self.df = self.df.pivot_table(
            values=values_column,
            index=[
                c
                for c in self.df.columns
                if c not in [feature_names_column, values_column]
            ],
            columns=feature_names_column,
        )
        self.df.reset_index(inplace=True)
        self.df.columns.name = ""

    def groupby(self, by: Union[str, List[str]]):
        """Returns multiple new Dataset objects by splitting on columns

        Akin to performing groupby on a pd.DataFrame, split a dataset on one or many
        columns and return a list of Phenonaut Datasets containing the information
        contained within each unique split.

        Parameters
        ----------
        by : Union[str, list[str]]
            If a string, then this is used as a column name upon which to group the
            dataset and return unique classes based on this column.  A list of strings
            is also allowed, enabling grouping of datasets by multiple columns, such as
            ['timepoint', 'concentration']

        Returns
        -------
        List[phenonaut.Dataset]
            A list of new phenonaut.Dataset objects split on the value(s) of the by
            argument
        """
        metadata = self._metadata
        metadata["initial_features"] = self.features
        if isinstance(by, str):
            by = [by]
        new_datasets = [
            Dataset(
                f"{self.name}_groupby_{','.join(by)}_{','.join([str(item) for item in index]) if not isinstance(index, str) else str(index)}",
                df,
                metadata=self._metadata,
            )
            for index, df in self.df.groupby(by)
        ]
        for ds in new_datasets:
            ds.history.append(
                TransformationHistory(
                    ds.features,
                    f"Dataset split with groupby on {by}, and taking value(s) {[ds.df[b].unique()[0] for b in by]}",
                )
            )
        return new_datasets

    def new_aggregated_dataset(
        self,
        identifier_columns: list[str],
        new_dataset_name: str = "Merged rows dataset",
        transformation_lookup: dict[str, Union[Callable, str]] = None,
        tranformation_lookup_default_value: Union[str, Callable] = "mean",
    ):
        """Merge dataset rows and make a new dataframe

        If we have a pd.DataFrame containing data derived from 2 fields of view
        from a microscopy image, a sensible approach is averaging features.
        If we have the DataFrame below, we may merge FOV 1 and FOV 2, taking the
        mean of all features.  As strings such as filenames should be kept, they
        are concatenated together, separated by a comma, unless the strings are
        the same, in which case just one is used.

        Here we test a df as follows:

        === ======= ======= ======= ======= ======= =========== ===
        ROW COLUMN  BARCODE feat_1  feat_2  feat_3  filename    FOV
        === ======= ======= ======= ======= ======= =========== ===
        1   1       Plate1  1.2	    1.2	    1.3	    fileA.png   1
        1   1       Plate1  1.3	    1.4	    1.5	    FileB.png   2
        1   1       Plate2  5.2	    5.1	    5	    FileC.png   1
        1   1       Plate2  6.2	    6.1	    6.8	    FileD.png   2
        1   2       Plate1  0.1	    0.2	    0.3	    fileE.png   1
        1   2       Plate1  0.2	    0.2	    0.38    FileF.png   2
        === ======= ======= ======= ======= ======= =========== ===


        Merging produces:

        ==== ====== ======== ======= ======= ====== =================== =====
        ROW  COLUMN BARCODE  feat_1  feat_2  feat_3            filename   FOV
        ==== ====== ======== ======= ======= ====== =================== =====
        1       1   Plate1    1.25     1.3    1.40  fileA.png,FileB.png   1.5
        1       1   Plate2    5.70     5.6    5.90  FileC.png,FileD.png   1.5
        1       2   Plate1    0.15     0.2    0.34  FileF.png,fileE.png   1.5
        ==== ====== ======== ======= ======= ====== =================== =====

        Note that the FOV column has also been averaged.

        Parameters
        ----------
        identifier_columns : list[str]
            If a biochemical assay evaluated through imaging is identified by a
            row, column, and barcode (for the plate) but multiple images taken
            from a well, then these multiple fields of view can be merged,
            creating averaged features.
        new_dataset_name : str, optional
            Name for the new Dataset, by default "Merged rows dataset"
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

        Returns
        -------
        Dataset
            Dataset with samples merged.
        """

        if transformation_lookup is None:
            transformation_lookup = {
                np.dtype("O"): lambda x: ",".join([f"{item}" for item in set(x)])
            }
        dtypes_dict = {n: dt for n, dt in zip(self.df.columns, self.df.dtypes)}
        transforms = {
            c: transformation_lookup.get(dt, tranformation_lookup_default_value)
            for c, dt in zip(self.df.columns, self.df.dtypes)
        }
        merged_df = (
            self.df.copy()
            .groupby(identifier_columns, as_index=False)
            .aggregate(transforms)
        )

        for id_col_name in identifier_columns:
            merged_df[id_col_name] = merged_df[id_col_name].astype(
                dtypes_dict[id_col_name]
            )
        tmp_metadata = self._metadata.copy()
        if "features_prefix" in tmp_metadata:
            del tmp_metadata["features_prefix"]
        if "initial_features" in tmp_metadata:
            del tmp_metadata["initial_features"]
        tmp_metadata["features"] = self.features

        new_ds = Dataset(
            dataset_name=new_dataset_name,
            input_file_path_or_df=merged_df,
            metadata=tmp_metadata,
        )
        new_ds._history = self.get_history()
        new_ds.features = (
            self.features,
            f"Merged rows of the dataset, using columns {identifier_columns}",
        )
        return new_ds

    def __update_hash(self):
        self.sha256.update(self.name.encode("utf-8"))
        self.sha256.update(
            ";".join(
                f"{','.join(feat)}:{t_hist}" for feat, t_hist in self.get_history()
            ).encode("utf-8")
        )
        self.sha256.update("".join(self._metadata).encode("utf-8"))
        self.sha256.update(",".join(self.features).encode("utf-8"))
        if self.df is not None:
            self.sha256.update(self.df.values.tobytes())
            self.sha256.update(self.df.columns.values.tobytes())

    def remove_features_with_outliers(
        self, outlier_cutoff=15.0, remove_data: bool = False
    ):
        """Removes feature columns containing values greater than given cutoff

        By default, any feature containing a value greater than 15 is removed. This cutoff can
        be raised and lowered as appropriate.

        Parameters
        ----------
        outlier_cutoff : float, optional
            If a feature column contains a value greater than this cutoff,
            then the feature is removed. By default 15.
        remove_data : bool, optional
            If True, then not only are feature columns with outliers removed from
            the Datasets list of features, but these columns are dropped from the
            DataFrames. If False, then only the Datasets list of features are
            changed. By default False.
        """
        outlier_features = self.data.columns[
            self.data.apply(lambda x: x.abs().max() >= outlier_cutoff)
        ].tolist()
        if remove_data:
            self.drop_columns(
                outlier_features,
                reason=f"outlier feature columns, cutoff={outlier_cutoff}",
            )
        else:
            self.features = (
                [f for f in self.features if f not in outlier_features],
                f"Dropped columns ({outlier_features}) outlier feature columns, cutoff={outlier_cutoff}",
            )

    def remove_blocklist_features(
        self,
        blocklist: Union[Path, str, list[str]],
        skip_first_line_in_file: bool = True,
        erase_data: bool = True,
        apply_to_non_features: bool = True,
        remove_prefixed: bool = True,
    ):
        r"""Remove blocklisted features/columns from a Dataset

        Allows removal of predefined feature blocklists.  Featurisation may generate
        features which are to be excluded from analysis as standard. This is the case
        with cellular images featurised with cell profiler. As such, there are a set
        of blocklist feautures which are often applied. This function allows
        specification of a list of features for removal (in the form of a list),
        or a string or path object denoting the location of a file containing this
        information.  A special string may also be passed to this function:
        "CellProfiler", which instructs Phenonaut to download the standard
        blocklist located here: https://figshare.com/ndownloader/files/23661539.
        Whilst matching features are removed, by default features which have a
        prefix on a blocklist matched feature are also removed. See parameters.

        Note: matching columns which are not features are also removed by
        default, see parameters.

        Parameters
        ----------
        blocklist : Union[Path, str, list[str]]
            A str or Path directing Phenonaut to where a text file of blocklisted
            features is stored.  Alternatively, a list of blocklisted features may
            be supplied. A special value is also accepted, whereby a string of
            "CellProfiler" is passed in, causing Phenonaut to retrieve the
            commonly used CellProfiler blocklisted features from
            https://figshare.com/ndownloader/files/23661539 .
        skip_first_line_in_file : bool, optional
            Commonly, blocklist files have a title line, which can be ignored
            before starting to list features. If True, then the first line is
            ignored. By default True.
        erase_data: bool
            If False, then no removal of columns from the Dataset is performed,
            only ensuring that no features are set which match the blocklist.
            This means that blocklist columns could persist in the Dataset as
            non-features. If True, then features are removed, and matching
            columns deleted. If False, apply_to_non_features has no effect.
            By default, True.
        apply_to_non_features: bool
            If True, then apply the filtering to columns as well as features. By
            default True.
        removed_prefixed: bool
            If True, features/columns may still be matched with blocklist
            features if they have a prefix followed by an underscore character.
            This allows transformations to be performed and features still
            removed. For example, applying the RobustMAD trasform prefixes
            features with 'RobustMAD\_', generating RobustMAD_FeatureA,
            RobustMAD_FeatureB etc.
            remove_blocklist_features will identify FeatureA (if in blocklist)
            and still remove that blocklisted feature. To deactivate this
            default behavior, set remove_prefixed_features to False. By default
            True.

        Raises
        ------
        FileNotFoundError
            Error raised if specified file is not found
        """

        if isinstance(blocklist, str):
            if blocklist == "CellProfiler":
                import requests

                blocklist = [
                    blf
                    for blf in requests.get(
                        "https://figshare.com/ndownloader/files/23661539"
                    )
                    .content.decode()
                    .split("\n")[1:]
                    if len(blf) > 1
                ]
            else:
                blocklist = Path(blocklist)
        if isinstance(blocklist, Path):
            if not blocklist.exists():
                raise FileNotFoundError(
                    f"Specified blocklist file {blocklist} does not exist"
                )
            with open(blocklist) as f:
                blocklist = [line.rstrip() for line in f]
            if skip_first_line_in_file:
                blocklist = blocklist[1:]

        if not erase_data:
            # Only removing features
            if remove_prefixed:
                features_to_keep = [
                    f
                    for f in self.features
                    if f not in blocklist
                    and not any([f.endswith(f"_{b}") for b in blocklist])
                ]
            else:
                features_to_keep = [f for f in self.features if f not in blocklist]
            self.features = (
                features_to_keep,
                f"Removed blocklist features ({list(set(self.features)-set(features_to_keep))})",
            )
            return
        else:
            if apply_to_non_features:
                if remove_prefixed:
                    to_remove = [
                        c
                        for c in self.df.columns
                        if c in blocklist
                        or any([c.endswith(f"_{b}") for b in blocklist])
                    ]
                else:
                    to_remove = [c for c in self.df.columns if c in blocklist]
            else:
                if remove_prefixed:
                    to_remove = [
                        f
                        for f in self.df.columns
                        if f in blocklist
                        or any([f.endswith(f"_{b}") for b in blocklist])
                    ]
                else:
                    to_remove = [f for f in self.df.columns if f in blocklist]
            self.filter_columns(to_remove, keep=False)

    def remove_low_variance_features(self, freq_cutoff=0.05, unique_cutoff=0.01):
        """Exclude low information content features.

        Adapted from pycytominer variance_threshold method
        https://github.com/cytomining/pycytominer/blob/master/pycytominer/operations/variance_threshold.py

        Sometimes, features can vary very little, this allows definition of cutoffs (ratios) of unique values
        that can exist in a feature. See parameters for further description of cutoffs.


        Parameters
        ----------
        freq_cutoff : float, default 0.05
            Ratio as defined by 2nd most common feature value divided by the most
            common feature value). Must range between 0 and 1.  Features below this
            cutoff have a large population with a unique value and will be removed.
        unique_cutoff: float, default 0.01
            Remove features with little diversity in their measurements.
            Must range between 0 and 1. Dividing the number of unique values in a
            feature by the number of measurements returns a 'unique' ratio, values
            below this cutoff are removed.

        """

        if not 0 <= freq_cutoff <= 1.0:
            raise ValueError("freq_cutoff must be greater than 0 and less than 1")
        if not 0 <= unique_cutoff <= 1:
            raise ValueError("unique_cutoff must be greater than 0 and less than 1")

        def _violates_frequency_cutoff(s, freq_cut):
            val_count = s.value_counts()
            try:
                if (val_count.iloc[1] / val_count.iloc[0]) < freq_cut:
                    return True
            except IndexError:
                return True
            return False

        # Subset dataframe

        # Features containing massively overrepresented common values
        features_to_remove_freq = self.data.apply(
            lambda x: _violates_frequency_cutoff(x, freq_cutoff)
        )
        features_to_remove_freq = features_to_remove_freq[
            features_to_remove_freq
        ].index.tolist()

        # Features with values too common
        unique_ratio = self.data.nunique() / self.df.shape[0]
        features_to_remove_unique = unique_ratio[
            unique_ratio < unique_cutoff
        ].index.tolist()

        self.drop_columns(
            list(set(features_to_remove_freq + features_to_remove_unique)),
            reason=": filtered low variance features",
        )

    def drop_nans_with_cutoff(
        self, axis: Optional[int] = None, nan_cutoff: float = 0.1
    ) -> None:
        """
        Drop rows or columns containing NaN or Inf values above a specified cutoff percentage.

        Parameters:
        -----------
        axis: Optional[int], default=None
            Axis along which to drop NaN or Inf values. If None, both rows and columns are dropped.
        nan_cutoff: float, default=0.1
            Cutoff percentage for NaN or Inf values. Rows or columns with NaN or Inf percentages greater than this
            value will be dropped.

        """

        # Compute the percentage of NaNs and Infs in each row and column
        row_nans_infs = self.data.isna().sum(axis=1) + np.isinf(self.data).sum(axis=1)
        col_nans_infs = self.data.isna().sum(axis=0) + np.isinf(self.data).sum(axis=0)
        row_pct = row_nans_infs / self.data.shape[1]
        col_pct = col_nans_infs / self.data.shape[0]

        if axis == 0:
            rows_to_drop = list(self.data.loc[(row_pct >= nan_cutoff), :].index)
            self.drop_rows(rows_to_drop)
        elif axis == 1:
            columns_to_drop = list(self.data.loc[:, col_pct >= nan_cutoff].columns)
            self.drop_columns(
                columns_to_drop,
                reason=f": filtered out features with large nan values above {nan_cutoff}",
            )
        else:
            rows_to_drop = list(self.data.loc[(row_pct >= nan_cutoff), :].index)
            self.drop_rows(rows_to_drop)
            columns_to_drop = list(self.data.loc[:, col_pct >= nan_cutoff].columns)
            self.drop_columns(
                columns_to_drop,
                reason=f": filtered out features with large nan values above {nan_cutoff}",
            )

    def impute_nans(
        self,
        groupby_col: Optional[Union[str, list[str]]] = None,
        impute_fn: Union[Callable, str, None] = "median",
    ) -> None:
        """
        Impute missing values in the DataFrame.

        Parameters:
        -----------
        groupby_col: str or list of str, default=None
            The name(s) of the column(s) to group by when imputing missing values.
            If None, impute missing values across the entire DataFrame.
        impute_fn: Union[Callable, str, None]
            The callable to use for imputing missing values on the DataFrame or
            grouped DataFrame as defined by the groupby_col. Special cases exist
            for 'median' and 'mean', whereby pd.median and pd.mean are applied.
            If None, then no action is taken. By default 'median'.
        """

        if impute_fn is None:
            return
        if impute_fn == "median":
            impute_fn = lambda df: df.fillna(df.median())

        elif impute_fn == "mean":
            impute_fn = lambda df: df.fillna(df.mean())

        if groupby_col is None:
            group = self.df[self.features]
        else:
            if not isinstance(groupby_col, list):
                groupby_col = [groupby_col]
            group = self.df[self.features + groupby_col]

        group = group.replace([np.inf, -np.inf], np.nan)

        if groupby_col is None:
            imputed_group = impute_fn(group)
            self.df[self.features] = imputed_group

        else:
            imputed_groups = group.groupby(groupby_col, group_keys=False).apply(
                impute_fn
            )
            self.df[self.features + groupby_col] = imputed_groups

        if groupby_col is not None:
            self.features = (
                self.features,
                f"Removed invalid entries by nan and inf handling on group column {groupby_col}",
            )
        else:
            self.features = (
                self.features,
                "Removed invalid entries by nan and inf handling",
            )

    def get_df_features_perturbation_column(
        self, quiet: bool = False
    ) -> tuple[pd.DataFrame, list[str], Union[str, None]]:
        """Helper function to obtain DataFrame, features and perturbation column name.

        Some Phenonaut functions allow passing of a Phenonaut object, or DataSet. They
        then access the underlying pd.DataFrame for calculations. This helper function
        is present on Phenonaut objects and Dataset objects, allowing more concise
        code and less replication when obtaining the underlying data.

        Parameters
        ----------
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
            self.df,
            self.features,
            self._metadata.get("perturbation_column", None)
            if quiet
            else self.perturbation_column,
        )
