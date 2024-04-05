# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import re
from pathlib import Path
from typing import Union

from pandas.errors import DataError

import phenonaut
from phenonaut.transforms.preparative import RemoveHighlyCorrelated
from phenonaut.utils import load_dict


class Workflow:
    """Phenonaut Workflows allow operation through simple YAML workflows.

    Workflows may be defined in Phenonaut, and the module executed directly,
    rather than imported and used by a Python program. Workflows are
    defined using the simple YAML file format. However, due to the way in
    which they are read in, JSON files may also be used. As YAML files can
    contain multiple YAML entries, we build on this concept, allowing
    multiple workflows to be defined in a single YAML (or JSON) file. Once
    read in, workflows are dictionaries. From Python 3.6 onwards,
    dictionaries are ordered. We can therefore define our workflows in order
    and guarantee that they will be executed in the defined order. A
    dictionary defining workflows have the following structure:
    {job_name: task_list}, where job_name is a string, and task list is a
    list defining callable functions, or tasks required to complete the job.

    The job list takes the form of a list of dictionaries, each containning
    only one key which is the name of the task to be peformed. The value
    indexed by this key is a dictionary of argument:value pairs to be passed
    to the function responsible for performing the task. The structure is
    best understood with an example. Here, we see a simple workflow
    contained within a YAML file for calculation of the scalar projection
    phenotypic metric. YAML files start with 3 dashes.

    .. code-block:: console

        ---
        scalar_projection_example:
        - load:
            file: screening_data.csv
            metadata:
                features_prefix:
                    - feat_
        - scalar_projection:
            target_treatment_column_name: control
            target_treatment_column_value: pos
            output_column_label: target_phenotype
        - write_multiple_csvs:
            split_by_column: PlateID
            output_dir: scalar_projection_output


    The equivalent JSON with clearer (for Python programmers) formatting for the above is:

    .. code-block:: python

        {
            "scalar_projection_example": [
                {
                    "load": {
                    "file": "screening_data.csv",
                    "metadata": {
                    "features_prefix": ["feat_"]}
                    }
                },
                {
                    "scalar_projection": {
                        "target_treatment_column_name": "control",
                        "target_treatment_column_value": "pos",
                        "output_column_label": "target_phenotype",
                    }
                },
                {"write_multiple_csvs":{
                    "split_by_column": "PlateID",
                    "output_dir": "scalar_projection_output/"
                    }
                },
            ]
        }


    The workflow define above in the example YAML and JSON formats, has the
    name "scalar_projection_example", and consists of 3 commands.

    #. load
    #. scalar_projection
    #. write_multiple_csvs

    See the user guide for a full listing of commands.

    Parameters
    ----------
    workflow_path : Union[Path, str, dict]
        Workflows can be defined in YML or JSON files with their locations
        supplied or jobs passed as dictionaries. Dictionary keys denote the
        job names. Values under these keys should be lists of dictionaries.
        Each dictionary should have one key, denoting the name of the task
        and values under this key contain options for the called functions/
        tasks.

    Raises
    ------
    TypeError
        Supplied Path or str to file location does not appear to be a YAML or JSON
        file.
    """

    def __init__(self, workflow_path: Union[Path, str, dict]):
        self.workflow = load_dict(workflow_path)
        self.run_workflow()

    def run_workflow(self):
        """Run the workflow defined in the workflow object instance"""

        if self.workflow is None:
            raise ValueError("Workflow not loaded")
        # Iterate jobs defined in the YAML file. Typically only one job will be defined,
        # but multiple are allowed.
        for job in self.workflow:
            self.phe = phenonaut.Phenonaut(name=job)
            # Jobs are built by
            for task in self.workflow[job]:
                if isinstance(task, str):
                    task = {task: {}}
                if len(task.keys()) > 1:
                    raise KeyError(
                        "Phenonaut workflow tasks require one top level dictionary entry - like 'load' or 'filter_columns'"
                    )
                command = next(iter(task))
                arguments = task[command]
                print(f"Performing : {command}")
                try:
                    getattr(self, command)(arguments)
                except AttributeError as error:
                    raise AttributeError(
                        f"{command} not implemented in Phenonaut Workflow"
                    )

    def load(self, arguments: dict):
        """Workflow function: load a dataset (CSV or PakcagedDataset)

        Workflow runnable function allowing loading of dataset from CSV or a
        PackagedDataset. As with all workflow runnable functions, this is
        designed to be called from a worflow.

        There are 2 possible options for loading in a dataset.

        Firstly, loading a user supplied CSV file.
        This option is initiated through inclusion of the 'file' key within
        arguments. The value under the 'file' key should be a string or Path to
        the CSV file.  In addition, a 'metadata' key is also required to be
        present in arguments, with a dictionary as the value. Within this
        dictionary under 'metadata', special keywords allow the reading of data
        in different formats. Special keys for the metadata dictionary are
        listed below (See Pandas documentation for read_csv for more in-depth
        information):

        sep
            define separator for fields within the file (by default ',')

        skiprows
            define a number of rows to skip at the beginning of a file

        header_row_number
            may be a single number, or list of numbers denoting header rows.

        transpose
            In the case of some transcriptomics raw data,
            transposition is required to have samples row-wise. Therefore the
            table must be transposed.  Set to True to transpose.

        index_col
            Column to use as the row labels of the CSV given as either
            the column names or as numerical indexes. Note: index_col=False can
            be used to force pandas to not use the first column as the index,
            e.g. when you have a malformed file with delimiters at the end of
            each line.

        key
            If an h5 file is supplied, then this is the key to the underlying pandas
            dataframe.

        Secondly, we may load a packaged dataset, by including the 'dataset' key
        within the arguments dictionary. The value under this key should be one
        of the current packaged datasets supported by workflow mode - currently
        TCGA, CMAP, Iris, Iris_2_views, BreastCancer. An additional key in the
        dictionary can be 'working_dir' with the value being a string denoting
        the location of the file data on a local filesystem, or the location
        that it should be downloaded to and stored.

        Parameters
        ----------
        arguments : dict
            Dictionary containing file and metadata keys, see function
            description/help/docstring.
        """
        if "file" in arguments and "dataset" in arguments:
            raise KeyError(
                "Cannot have both file and dataset arguments present within a load command"
            )

        if "file" in arguments:
            if "metadata" not in arguments:
                message = "Workflow load called without appropriate metadata arguments"
                raise KeyError(message)
            file = arguments["file"]
            metadata = arguments["metadata"]
            if file.endswith(".tsv") or file.endswith(".tsv.gz"):
                if "sep" not in metadata:
                    metadata["sep"] = "\t"
            if file.endswith(".h5"):
                if "key" not in arguments:
                    raise KeyError(
                        "Workflow load called on an h5 file, without supplying a key argument"
                    )
                self.phe.load_dataset(
                    "Workflow loaded dataset",
                    file,
                    metadata=metadata,
                    h5_key=arguments["key"],
                )
            self.phe.load_dataset("Workflow loaded dataset", file, metadata=metadata)
            return
        if "dataset" in arguments:
            packaged_dataset_name = metadata["dataset"]
            working_dir = arguments.get("working_dir", None)
            if packaged_dataset_name == "TCGA":
                from phenonaut.packaged_datasets import TCGA

                self.phe = phenonaut.Phenonaut(TCGA(working_dir))
                return
            if packaged_dataset_name == "CMAP":
                from phenonaut.packaged_datasets import CMAP

                self.phe = phenonaut.Phenonaut(CMAP(working_dir))
                return
            if packaged_dataset_name == "Iris":
                from phenonaut.packaged_datasets import Iris

                self.phe = phenonaut.Phenonaut(Iris(working_dir))
                return
            if packaged_dataset_name == "Iris_2_views":
                from phenonaut.packaged_datasets import Iris_2_views

                self.phe = phenonaut.Phenonaut(Iris_2_views(working_dir))
                return
            if packaged_dataset_name == "BreastCancer":
                from phenonaut.packaged_datasets import BreastCancer

                self.phe = phenonaut.Phenonaut(BreastCancer(working_dir))
                return
            raise KeyError(
                "Name of packaged dataset not found or accessible within a workflow"
            )
        raise KeyError("Did not find file or dataset argument for load command")

    def filter_rows(self, arguments: dict):
        """Workflow function: Filter rows

        Designed to be called from a workflow, filter_rows alows keeping of only
        rows with a certain value in a certain column. Takes as arguments a
        dictionary containing query_column key:value pair and one of
        query_value, query_values or values key:value pairs:

            query_column
                name of the column that should match the value below

            query_value
                value to match

            query_values
                values to match (as a list)

            values
                values to match (as a list)

        Additionally, a key "keep" with a boolean value may be included. If True
        then rows matching the query are kept, if False, then rows matching are
        discarded, and non-matching rows kept.

        Parameters
        ----------
        arguments : dict
            Dictionary containing query_column key and value defining the column
            name, and one of the following keys: query_value, query_values,
            values. If plural, then values under the key should be a list
            containing  values to perform matching on, otherwise, singular
            value.

        Raises
        ------
        DataError
            [description]
        """
        values = []
        for key in ["query_value", "query_values", "values"]:
            if key in arguments:
                if isinstance(arguments[key], list):
                    values.extend(arguments[key])
                else:
                    values.append(arguments[key])
        if values == []:
            raise DataError("No query_value or query_values found to filter on")
        keep = arguments["keep"]
        query_column = arguments["query_column"]
        ## filter here
        self.phe.ds.filter_rows(query_column, values, keep=keep)

    def filter_columns(self, arguments: dict):
        """Workflow function: filter columns

        Designed to be called from a workflow, Datasets may have columns
        defined for keeping or removal. This function also provides a convenient
        way to reorder dataframe columns.

        Parameters
        ----------
        arguments : dict
            Dictionary of options, can include the following keys:
                keep: bool, optional, by default True
                    Only matching columns are kept if true.  If false, they are
                    removed.
                column_names: [list, str]
                    List of columns to keep (or regular expressions to match)
                column_name: str
                    Singular column to keep (or regular expressions to match)
                regex: bool, optional, by default False.
                    perform regular expression matching

        """
        keep = arguments.pop("keep", None)
        use_regex = arguments.pop("regex", False)
        if keep is None:
            message = "filter_columns in a workflow must have a keep argument, taking values True (to keep these columns), or False (to remove the columns)"
            raise KeyError(message)

        values = []
        for key in [
            "column_name",
            "column_names",
            "values",
            "column_value",
            "column_values",
        ]:
            if key in arguments:
                if isinstance(arguments[key], list):
                    values.extend(arguments[key])
                else:
                    values.append(arguments[key])
        if values == []:
            raise DataError("No query_value or query_values found to filter on")
        ## filter here
        self.phe.ds.filter_columns(values, keep=keep, regex=use_regex)
        return

    def write_multiple_csvs(self, arguments: dict):
        """Workflow function: Write multiple CSV files

        Designed to be called from a workflow. Often it is useful to write a CSV
        file per plate within the dataset, or group the data by some other
        identifier.

        Parameters
        ----------
        arguments : dict
            Dictionary, should contain:

            split_by_column: str
                the column to be split on

            output_dir: str
                the target output directory

            file_prefix: str
                optional prefix for each file.

            file suffix: str
                optional suffix for each file.

            file_extension: str
                optional file extension, by default '.csv'

        """
        split_by_column = arguments.pop("split_by_column")
        output_dir = arguments.pop("output_dir", Path.cwd())
        file_prefix = arguments.pop("file_prefix", "")
        file_suffix = arguments.pop("file_suffix", "")
        file_extension = arguments.pop("file_extension", ".csv")
        self.phe.ds.df_to_multiple_csvs(
            split_by_column,
            output_dir=output_dir,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            file_extension=file_extension,
            **arguments,
        )

    def write_csv(self, arguments: dict):
        """Workflow function: Write dataframe to CSV file.

        Designed to be called from a workflow, writes a CSV file using the
        Pandas.DataFrame.to_csv function.  Expects a dictionary as arguments
        containing a 'path' key, with a string value pointing at the destination
        location of the CSV file. Additional keys within the supplied dictionary
        are supplied to Pandas.DataFrame.to_csv as kwargs, allowing fully
        flexible output.

        Parameters
        ----------
        arguments : dict
            Dictionary should contain a 'path' key, may also contain a target_dataset key
            which if absent, defaults to -1 (usually the last added dataset).
        """
        file_path = arguments.pop("path")
        self.phe[arguments.get("target_dataset", -1)].df_to_csv(file_path, **arguments)

    def scalar_projection(self, arguments: dict):
        """Workflow function: Add a column for the scalar projection to a target perturbation.

        Designed to be called from a workflow, calculates the scalar projection
        and scalar rejection, quantifying on and off target phenotypes, as
        used in:
        Heiser, Katie, et al. "Identification of potential treatments for
        COVID-19 through artificial intelligence-enabled phenomic analysis
        of human cells infected with SARS-CoV-2." BioRxiv (2020).

        Parameters
        ----------
        arguments : dict, should contain:
            target_dataset
                Index or name of dataset which should be used in the measurment.
                If absent, then behaviour is as if -1 is supplied, indicating
                the last added dataset.
            target_perturbation_column_name
                normally a 'control' column
            target_perturbation_column_value:
                value to be found in the column defined previously.
            output_column_label:
                Output from the scalar projection will have the form: on_target_<output_column_label> and off_target_<output_column_label>
                if this is missing, then it is set to target_perturbation_column_value

        """
        if "output_column_label" not in arguments:
            arguments["output_column_label"] = arguments[
                "target_perturbation_column_value"
            ]
        phenonaut.metrics.scalar_projection(
            self.phe[arguments.get("target_dataset", -1)], **arguments
        )

    def euclidean_distance(self, arguments: dict):
        """Workflow function: Add a column for the euclidean distance to a target perturbation.

        Designed to be called from a workflow, calculates the euclidean distance
        in feature space.

        Parameters
        ----------
        arguments : dict, should contain:
            target_dataset
                Index or name of dataset which should be used in the measurment.
                If absent, then behaviour is as if -1 is supplied, indicating
                the last added dataset.
            target_perturbation_column_name
                normally a 'control' column
            target_perturbation_column_value:
                value to be found in the column defined previously.
            output_column_label:
                Output column for the measurement. If this is missing, then it is set to target_perturbation_column_value.

        """
        if "output_column_label" not in arguments:
            arguments["output_column_label"] = arguments[
                "target_perturbation_column_value"
            ]
        target_perturbation_column_name = arguments["target_perturbation_column_name"]
        target_perturbation_column_value = arguments["target_perturbation_column_value"]
        output_column_label = arguments["output_column_label"]
        ds = self.phe[arguments.get("target_dataset", -1)]
        ds.df[output_column_label] = phenonaut.metrics.euclidean(
            ds.data,
            ds.df.query(
                f"{target_perturbation_column_name} == '{target_perturbation_column_value}'"
            ),
        )

    def mahalanobis_distance(self, arguments: dict):
        """Workflow function: Add a column for the Mahalanobis distance to target perturbations.

        Designed to be called from a workflow, calculates the Mahalanobis distance
        in feature space.

        Parameters
        ----------
        arguments : dict, should contain:
            target_dataset
                Index or name of dataset which should be used in the measurment.
                If absent, then behaviour is as if -1 is supplied, indicating
                the last added dataset.
            target_perturbation_column_name
                normally a 'control' column
            target_perturbation_column_value:
                value to be found in the column defined previously.
            output_column_label:
                Output column for the measurement. If this is missing, then it is set to target_perturbation_column_value.

        """
        if "output_column_label" not in arguments:
            arguments["output_column_label"] = arguments[
                "target_perturbation_column_value"
            ]
        target_perturbation_column_name = arguments["target_perturbation_column_name"]
        target_perturbation_column_value = arguments["target_perturbation_column_value"]
        output_column_label = arguments["output_column_label"]
        ds = self.phe[arguments.get("target_dataset", -1)]
        ds.df[output_column_label] = phenonaut.metrics.mahalanobis(
            ds.data,
            ds.df.query(
                f"{target_perturbation_column_name} == '{target_perturbation_column_value}'"
            ),
        )

    def manhattan_distance(self, arguments: dict):
        """Workflow function: Add a column for the Manhattan distance to target perturbation.

        Designed to be called from a workflow, calculates the Manhattan distance
        in feature space. Also known as the cityblock distance.

        Parameters
        ----------
        arguments : dict, should contain:
            target_dataset
                Index or name of dataset which should be used in the measurment.
                If absent, then behaviour is as if -1 is supplied, indicating
                the last added dataset.
            target_perturbation_column_name
                normally a 'control' column
            target_perturbation_column_value:
                value to be found in the column defined previously.
            output_column_label:
                Output column for the measurement. If this is missing, then it is set to target_perturbation_column_value.

        """
        if "output_column_label" not in arguments:
            arguments["output_column_label"] = arguments[
                "target_perturbation_column_value"
            ]
        target_perturbation_column_name = arguments["target_perturbation_column_name"]
        target_perturbation_column_value = arguments["target_perturbation_column_value"]
        output_column_label = arguments["output_column_label"]
        ds = self.phe[arguments.get("target_dataset", -1)]
        ds.df[output_column_label] = phenonaut.metrics.manhattan(
            ds.data,
            ds.df.query(
                f"{target_perturbation_column_name} == '{target_perturbation_column_value}'"
            ),
        )

    def cityblock_distance(self, arguments: dict):
        """Workflow function: Add a column for the cityblock distance to a target perturbation.

        Designed to be called from a workflow, calculates the cityblock distance
        in feature space. Also known as the Manhattan distance.

        Parameters
        ----------
        arguments : dict, should contain:
            target_dataset
                Index or name of dataset which should be used in the measurment. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
            target_perturbation_column_name
                normally a 'control' column
            target_perturbation_column_value:
                value to be found in the column defined previously.
            output_column_label:
                Output column for the measurement. If this is missing, then it is set to target_perturbation_column_value.

        """
        if "output_column_label" not in arguments:
            arguments["output_column_label"] = arguments[
                "target_perturbation_column_value"
            ]
        target_perturbation_column_name = arguments["target_perturbation_column_name"]
        target_perturbation_column_value = arguments["target_perturbation_column_value"]
        output_column_label = arguments["output_column_label"]
        ds = self.phe[arguments.get("target_dataset", -1)]
        ds.df[output_column_label] = phenonaut.metrics.manhattan(
            ds.data,
            ds.df.query(
                f"{target_perturbation_column_name} == '{target_perturbation_column_value}'"
            ),
        )

    def rename_column(self, arguments: dict):
        """Workflow function: Rename column

        Designed to be called from a workflow, ranames a single, or multiple
        columns. The arguments dictionary should contain key:value pairs, where
        the key is the old column name and the value is the new column name.

        Parameters
        ----------
        arguments : dict
            Dictionary containing name_from:name_to key value pairs, which will
            cause the column named 'name_from' to be renamed 'name_to'.
            Multiple columns can be renamed in a single call, using multiple
            dictionary entries.

        Raises
        ------
        ValueError
            'arguments' was not a dictionary of type: str:str
        """
        if isinstance(arguments, str):
            message = f"The argument to rename_column/rename_columns was a string, not a dictionary, if using a YAML workflow to define the job, make sure there is a space after the key - eg. 'oldkey: newkey', rather than 'oldkey:newkey'"
            raise ValueError(message)
        if len(arguments.keys()) > 0:
            self.phe.ds.rename_columns(arguments)

    def rename_columns(self, arguments: dict):
        """Workflow function: Rename columns

        Designed to be called from a workflow, ranames a single, or multiple
        columns. The arguments dictionary should contain key:value pairs, where
        the key is the old column name and the value is the new column name.

        Parameters
        ----------
        arguments : dict
            Dictionary containing name_from:name_to key value pairs, which will
            cause the column named 'name_from' to be renamed 'name_to'.
            Multiple columns can be renamed in a single call, using multiple
            dictionary entries.

        Raises
        ------
        ValueError
            'arguments' was not a dictionary of type: str:str
        """
        self.rename_column(arguments)

    def add_well_id(self, arguments: dict):
        """Workflow function: Add well IDs

        Designed to be called from a workflow. Often, we would like to use well
        and column numbers to resolve a more traditional alpha-numeric WellID
        notation, such as A1, A2, etc. This can be achieved through calling this
        workflow function.

        If a dataset contains numerical row and column names, then they may be
        translated into standard letter-number well IDs. The arguments
        dictionary may contain the following keys, with their values denoted
        as bellow:

        numerical_column_name : str, optional
            Name of column containing numeric column number, if not supplied,
            then behaves as if "COLUMN".
        numerical_row_name : str, optional
            Name of column containing numeric column number, if not supplied,
            then behaves as if "ROW".
        plate_type : int, optional
            Plate type - note, at present, only 384 well plate format is
            supported, if not supplied, then behaves as if 384.
        new_well_column_name : str, optional
            Name of new column containing letter-number well ID, if not
            supplied, then behaves as if default "Well".
        add_empty_wells : bool, optional
            Should all wells from a plate be inserted, even when missing from
            the data, if not supplied, then behaves as if False.
        plate_barcode_column : str, optional
            Multiple plates may be in a dataset, this column contains their
            unique ID, if not supplied, then bahaves as if None.
        no_sort : bool, optional
            Do not resort the dataset by well ID, if not supplied, then behaves
            as if False

        Parameters
        ----------
        arguments : dict
            Dictionary containing arguments to the Dataset.add_well_id function,
            see API documentation for further details, or function help.

        """
        self.phe.ds.add_well_id(**arguments)

    def copy_column(self, arguments: dict):
        """Workflow function: copy a dataset column

        Designed to be called from a workflow, copys the values of one column
        within a dataset to another. The arguments dictionary can contain 'to'
        and 'from' keys with values for column names, or alternatively, simply
        from:to key-value pairs denoting how to perform the copy operation.

        Parameters
        ----------
        arguments : dict
            Options for the command. Should include either:
                1.  dictionary with keys "to" and "from", with item names related to the
                columns that should be used.

                2.dictionary with the form {from_column:to_column}, which will copy the
                column with title from_column to to_column

            Note, if any dictionary items (to) are lists, then multiple copies will
            be made.

        Raises
        ------
        KeyError
            Column was not found in the Pandas DataFrame.
        """
        from_column = arguments.pop("from", None)
        to_column = arguments.pop("to", None)
        if any([from_column, to_column]):
            if not all([from_column, to_column]):
                message = "Only one of 'to' or 'from' found as keys, copy_columns needs a from and to argument, alternatively, give a dictionary key pair of colA:colB to copy colA to colB"
                raise KeyError(message)
            arguments[from_column] = to_column
        for from_column, to_column in arguments.items():
            if not isinstance(to_column, list):
                to_column = [to_column]
            for destination in to_column:
                print(f"Copying {from_column} to {destination}")
                self.phe.ds.df[destination] = self.phe.ds.df[from_column]

    def if_blank_also_blank(self, arguments: dict):
        """Workflow function: if column is empty, also blank

        Designed to be called from a workflow, often it is required to clean or
        remove rows not needed for inclusion into further established pipelines/
        workflows. This workflow function allows the ability to remove values
        from a column on the condition that onther column is empty.

        Parameters
        ----------
        arguments : dict
            Dictionary containing the following key:value pairs:

            query_column
                value is the name of the column to perform the query on.
            regex_query
                value is a boolean denoting if the query column value
                should be matched using a regular expression. If omitted, then
                behaves as if present and False.
            target_column
                value is a string, denoting the name of the column
                which should be blanked.
            target_columns
                value is a list of strings, denoting the names of columns which should be blanked.
            regex_targets
                value is a boolean denoting if the target column or multiple target columns defined in target_columns should be matched using a regular expression. If absent, then behaves as if False was supplied.

        Raises
        ------
        KeyError
            'query_column' not found in arguments dictionary
        IndexError
            Multiple columns matched query_column using the regex
        KeyError
            No target columns found for if_blank_also_blank, use target_column
            keys
        """

        query_column = arguments.pop("query_column", None)
        if query_column is None:
            message = f"if_blank_also_blank needs a 'query_column' argument"
            raise KeyError(message)
        if arguments.pop("regex_query", False):
            query_column = list(
                filter(re.compile(query_column).match, self.phe.df.columns.values)
            )
            if len(query_column) == 1:
                query_column = query_column[0]
            else:
                message = "Multiple columns matched query_column using the regex"
                raise IndexError(message)

        target_columns = []
        for tc in ["target_column", "target_columns"]:
            if tc in arguments:
                tmp_target_cols = arguments.pop(tc)
                if isinstance(tmp_target_cols, str):
                    tmp_target_cols = [tmp_target_cols]
                target_columns.extend(tmp_target_cols)
        if len(target_columns) == 0:
            message = "No target columns found for if_blank_also_blank, use target_column keys"
            raise KeyError(message)

        if arguments.pop("regex_targets", False):
            expanded_target_columns = []
            for pattern in target_columns:
                for match in list(
                    filter(re.compile(pattern).match, self.phe.df.columns.values)
                ):
                    if match not in expanded_target_columns:
                        expanded_target_columns.append(match)
            target_columns = expanded_target_columns

        print(f"{target_columns=}  {query_column}")
        self.phe.df.loc[self.phe.df[query_column].isnull(), target_columns] = ""

    def pca(self, arguments: dict):
        """Workflow function: Perform PCA dimensionality reduction technique.

        Designed to be called from a workflow, performs the principal component
        dimensionality reduction technique. If no arguments are given or the
        arguments dictionary is empty, then 2D PCA is applied to the dataset
        with the highest index, equivalent of phe[-1], which is usually the last
        inserted dataset.

        Parameters
        ----------
        arguments : dict
            Dictionary of arguments to used to direct the PCA process, can
            contain the following keys and values

            target_dataset
                Index or name of dataset which should have the
                dimensionality reduction applied. If absent, then
                behaviour is as if -1 is supplied, indicating the last added
                dataset.
            ndims
                Number of dimensions to which the PCA should reduce the
                features. If absent, then defaults to 2.
            center_on_perturbation_id
                PCA should be recentered on the
                perturbation with ID. If absent, then defaults to None, and
                no centering is performed.
            center_by_median
                If true, then median of center_on_perturbation
                is used, if False, then the mean is used.
            fit_perturbation_ids
                PCA may be fit to only the included IDs,
                before the transform is applied to the whole dataset.
        """
        ndims = arguments.get("ndims", 2)
        center_on_perturbation = arguments.get("center_on_perturbation", None)
        from phenonaut.transforms import PCA

        transformer = PCA()
        transformer(
            self.phe[arguments.get("target_dataset", -1)],
            arguments.get("ndims", 2),
            center_on_perturbation_id=arguments.get("center_on_perturbation_id"),
            center_by_median=arguments.get("center_by_median", True),
            fit_perturbation_ids=arguments.get("fit_perturbation_ids"),
        )

    def umap(self, arguments: dict):
        """Workflow function: Perform UMAP dimensionality reduction technique.

        Designed to be called from a workflow, performs the UMAP dimensionality
        reduction technique. If no arguments are given or the arguments
        dictionary is empty, then 2D UMAP is applied to the dataset with the
        highest index, equivalent of phe[-1], which is usually the last inserted
        dataset.

        Parameters
        ----------
        arguments : dict
            Dictionary of arguments used to direct the UMAP transform function.
            Can contain the following keys and values.

            target_dataset
                Index or name of dataset which should have the
                dimensionality reduction applied. If absent, then
                behaviour is as if -1 is supplied, indicating the last added
                dataset.
            ndims
                Number of dimensions to which the UMAP should reduce the
                features. If absent, then defaults to 2.
            center_on_perturbation_id
                UMAP should be recentered on the
                perturbation with ID. If absent, then defaults to None, and
                no centering is performed.
            center_by_median
                If true, then median of center_on_perturbation
                is used, if False, then the mean is used.
        """
        ndims = arguments.get("ndims", 2)
        center_on_perturbation = arguments.get("center_on_perturbation", None)
        from phenonaut.transforms import UMAP

        transofmer = UMAP()
        transofmer(
            self.phe[arguments.get("target_dataset", -1)],
            arguments.get("ndims", 2),
            center_on_perturbation_id=arguments.get("center_on_perturbation_id"),
            center_by_median=arguments.get("center_by_median", True),
        )

    def tsne(self, arguments: dict):
        """Workflow function: Perform t-SNE dimensionality reduction technique.

        Designed to be called from a workflow, performs the t-SNE dimensionality
        reduction technique. If no arguments are given or the arguments
        dictionary is empty, then 2D t-SNE is applied to the dataset with the
        highest index, equivalent of phe[-1], which is usually the last inserted
        dataset.



        Parameters
        ----------
        arguments : dict
            Dictionary of arguments to used to direct the t-SNE process, can
            contain the following keys and values:

            target_dataset
                Index or name of dataset which should have the
                dimensionality reduction applied. If absent, then
                behaviour is as if -1 is supplied, indicating the last added
                dataset.
            ndims
                number of dimensions to which the t-SNE should reduce
                the features. If absent, then defaults to 2.
            center_on_perturbation_id
                tSNE should be recentered on the
                perturbation with ID.If absent, then defaults to None, and
                no centering is performed.
            center_by_median
                If true, then median of center_on_perturbation
                is used, if False, then the mean is used.
        """
        ndims = arguments.get("ndims", 2)
        center_on_perturbation = arguments.get("center_on_perturbation", None)
        from phenonaut.transforms import TSNE

        transofmer = TSNE()
        transofmer(
            self.phe[arguments.get("target_dataset", -1)],
            arguments.get("ndims", 2),
            center_on_perturbation_id=arguments.get("center_on_perturbation_id"),
            center_by_median=arguments.get("center_by_median", True),
        )

    def set_perturbation_column(self, arguments: dict):
        """Workflow function: Set the perturbation column

        Designed to be called from a workflow, the perturbation column can be
        set on a dataset to help with plotting/scatters.

        Parameters
        ----------
        arguments : dict
            target_dataset:
                Index or name of dataset within which we wish to set the perturbation
                column. If absent, then behaviour is as if -1 is supplied, indicating
                the last added dataset.
            column:
                str type giving the new column name which will be set to mark
                perturbations.

        """
        if "column" not in arguments:
            raise KeyError("Must supply a 'column' key in arguments dictionary")
        self.phe[arguments.get("target_dataset", -1)].perturbation_column = arguments[
            "column"
        ]

    def VIF_filter_features(self, arguments: dict):
        """Workflow function: Perform VIF feature filter

        Designed to be called from a workflow, performs variance inflation
        factor (VIF) filtering on a dataset, removing features which are not
        detrimental to capturing variance.
        More information available:
        https://en.wikipedia.org/wiki/Variance_inflation_factor

        This can be a computationally expensive process as the number of
        linear regressions required to be run is almost N^2 with features.

        Parameters
        ----------
        arguments : dict
            - target_dataset:
                Index or name of dataset which should have variance
                inflation filter applied. If absent, then behaviour is as if -1
                is supplied, indicating the last added dataset.

            - vif_cutoff:
                float or int indicating the VIF cutoff to apply. A good
                balance and value often used is 5.0.  If this key:value pair is
                absent, then behaviour is as if 5.0 was supplied.

            - min_features:
                removal of too many features can be detrimental.
                Setting this value sets a lower limit on the number of features
                which must remain.  If absent, then behaviour is as if a value
                of 2 was given.

            - drop_columns:
                value is a boolean, denoting if columns should be
                dropped from the data table, as well as being removed from
                features. If not supplied, then the behaviour is as if False was
                supplied.


        """
        from phenonaut.transforms.preparative import VIF

        vif = VIF()
        ds_id = arguments.pop("target_dataset", -1)
        vif(self.phe[ds_id], **arguments)

    def filter_correlated_features(self, arguments: dict):
        """Workflow function: Perform filter of highly correlated features

        Designed to be called from a workflow, performs filtering of highly
        correlated features (as calculated by Pearson correlation coefficient)
        by either by removal of features correlated above a given theshold, or
        uses the iterative removal of features with the highest R^2 against
        another feature.  The arguments dictionary should contain a threshold or
        n key:value pair, not both.  A key of threshold and float value defines
        the correlation above which, features should be removed. If the n key is
        present, then features are iteratively removeduntil n features remain.

        Parameters
        ----------
        arguments : dict

            target_dataset
                Index or name of dataset which should have features
                filtered. If absent, then behaviour is as if -1 is supplied,
                indicating the last added dataset.
            threshold
                If this key is present, then it activates threshold
                mode, where by calculated correlations, above which should be
                removed. A good value for this threshold is 0.9.
            n
                If this key is present, then the number of features to keep is
                defined this way. The process works through iteratively removing
                features ordered by the most correlated until the number of
                features is equal to n. If threshold is also present, then n
                acts as a minimum number of features and feature removal will
                stop, no matter the correlations present in the dataset.
            drop_columns : bool, optional
                If drop columns is True, then not only will features be removed
                from the dataset features list, but the columns for these
                features will be removed from the dataframe. If absent, then the
                behaviour is as if False was supplied as a value to this
                key:value pair.

        """
        from phenonaut.transforms.preparative import RemoveHighlyCorrelated

        rhc = RemoveHighlyCorrelated()
        ds_id = arguments.pop("target_dataset", -1)
        rhc(self.phe[ds_id], **arguments)

    def filter_correlated_and_VIF_features(self, arguments: dict):
        """Workflow function: Filter features by highly correlated then VIF.

        Designed to be called from a workflow, Ideally, VIF would be applied to
        very large datasets.  Due to the almost n^2 number of linear regression
        required as features increase, this is not possible on datasets with a
        large number of features - such as methylation datasets.  We therefore
        must use other methods to reduce the features to a comfortable level
        allowing VIF to be performed. This class calculates correlations between
        all features and iteratively (Pearson correlation coefficient), removing
        features with the highest R^2 against another feature. Once the number
        of featurs is reduced to a level suitable for VIF, VIF is performed.

        More information available:
        https://en.wikipedia.org/wiki/Variance_inflation_factor

        Parameters
        ----------
        arguments : dict
            - target_dataset:
                Index or name of dataset which should have features
                filtered. If absent, then behaviour is as if -1 is supplied,
                indicating the last added dataset.
            - n_before_vif :
                Number of features to remove before applying VIF.
                This is required when dealing with large datasets which would be
                too time consuming to process entirely with VIF.
                Features are removed iteratively, selecting the most correlated
                features and removing them. If this key:value pair is absent,
                then it is as if the value of 1000 has been supplied.
            - vif_cutoff :
                The VIF cutoff value, above which features are
                removed. Features with VIF scores above 5.0 are considered
                highly correlated. If not supplied, then behaviour is as if a
                value of 5.0 was supplied.
            - drop_columns :
                If drop columns is True, then not only will features
                be removed from the dataset features list, but the columns for
                these features will be removed from the dataframe. If absent,
                then behaviour is as if False was supplied.


        """
        from phenonaut.transforms.preparative import RemoveHighestCorrelatedThenVIF

        rhctv = RemoveHighestCorrelatedThenVIF()
        ds_id = arguments.pop("target_dataset", -1)
        rhctv(self.phe[ds_id], **arguments)

    def scatter(self, arguments: dict):
        """Workflow function: Make scatter plot.

        Designed to be called from a workflow, produce a scatter plot from a
        dataframe.

        Parameters
        ----------
        arguments : dict

            target_dataset
                Index or name of dataset which should have features
                plotted. If absent, then behaviour is as if -1 is supplied,
                indicating the last added dataset.
            figsize
                A tuple denoting the target output size in inches (!), if
                absent, then the default of (8,6) is used.
            title
                Title for the plot, if absent, then the default "2D scatter"
                is used.
            peturbations
                Can be a list of peturbations - as denoted in the
                perturbations column of the dataframe to include in the plot.
                If absent, then all perturbations are included.
            destination
                Output location for the PNG - required field.  An error
                will be thrown if omitted.

        """

        if "destination" not in arguments:
            raise KeyError(
                "The key 'destination' with a location to write the output PNG must be present within the arguments dictionary"
            )
        from phenonaut.output import Scatter

        ds_id = arguments.get("target_dataset", -1)
        figsize = arguments.get("figsize", (8, 6))
        title = arguments.get("title", "2D scatter")
        perturbations = arguments.get("perturbations", None)

        destination = arguments.get("destination")
        scatter = Scatter(figsize=figsize, title=title)
        scatter.add(self.phe[ds_id], perturbations=perturbations)
        scatter.save_figure(destination)


def predict(self, arguments: dict):
    """Workflow function: predict


    Profile predictors in their ability to predict a given target.

    Phenonaut provides functionality to profile the performance of multiple predictors against
    multiple views of data. This is exemplified in the TCGA example used in the Phenonaut paper -
    see Example 1 - TCGA for a full walkthrough of applying this functionality to The Cancer Genome Atlas.
    With a given ‘target’ for prediction which is in the dataset, predict selects all appropriate predictors
    (classifiers for classification, regressors for regression and multiregressors for multi regression/view targets).
    Then, enumerating all views of the data and all predictors, hyperparameter optimisation coupled with 5-fold cross
    validation using Optuna is employed, before finally testing the best hyperparameter sets with retained test sets.
    This process is automatic and requires the data, and a prediction target. Output from this process is extensive
    and it may take a long time to complete, depending on the characteristics of your input data. Writen output from
    the profiling process consists of performance heatmaps highlighting best view/predictor combinations in bold,
    boxplots for each view combination and a PPTX presentation file allowing easy sharing of data, along with machine
    readable CSV and JSON results.

    For each unique view combination and predictor, perform the following:

        - Merge views and remove samples which do not have features across currently needed views.

        - Shuffle the samples.

        - Withhold 20% of the data as a test set, to be tested against the trained and hyperparameter optimised predictor.

        - Split the data using 5-fold cross validation into train and validation sets.

        - For each fold, perform Optuna hyperparameter optimisation for the given predictor using the train sets, using hyperparameters described by the default predictors for classification, regression and multiregression.

    Parameters
    ----------

    arguments : dict

        output_directory
            Directory into which profiling output (boxplots, heatmaps, CSV, JSON and PPTX should be written).

        dataset_combinations
            If multiple datasets are already loaded, then lists of ‘views’ may be specified for exploration. If None, or this argument is absent, then all combinations of available views/Datasets are enumerated and used.

        target
            The prediction target, denoted by a column name given here which exists in loaded datasets.

        n_splits
            Number of splits to use in the N-fold cross validation, if absent, then the default of 5 is used.

        n_optuna_trials
            Number of Optuna trials for hyperparameter optimisation, by default 20. This drastically impacts runtime, so if things are taking too long, you may wish to lower this number. For a more thorough exploration of hyperparameter space, increase this number.

        optuna_merge_folds
            By default, each fold has hyperparameters optimised and the trained predictor with parameters reported. If this optuna_merge_folds is true, then each fold is trained on and and hyperparameters optimised across folds (not per-fold). Setting this to False may be useful depending on the intended use of the predictor. It is believed that when False, and parameters are not optimised across folds, then more accurate prediction variance/accuracy estimates are produced. If absent, behaves as if false.

        test_set_fraction
            When optimising a predictor, by default a fraction of the total data is held back for testing, separate from the train-validation splits. This test_set_fraction controls the size of this split. If absent, then the default value of 0.2 is used.

    """

    if "destination" not in arguments:
        raise KeyError(
            "The key 'destination' with a location to write the output PNG must be present within the arguments dictionary"
        )
    from phenonaut.predict import profile

    predict(**arguments)
