#############
Workflow mode
#############

Phenonaut may be used in a workflow mode, removing the requirement for users to be able to use Python.


With the definition of workflows in simple YAML files (`<https://en.wikipedia.org/wiki/YAML/>`_), 
standardised workflows can be created that have access to all Phenonuat functionality. 
Whilst YAML files are a simple and convenient format, JSON dictionaries respresented as text files in
the .json format may also be passed. For the purposes of this documentation, the YAML format is used
to exemplify Phenonaut capabilities. Phenonaut will determine if it is working with YAML or JSON files
through the workflow filename extension.

The example YAML file for prediction of 1 year survival rates is as follows:

.. code-block:: yaml

    ---
    Example_1_predict_survival:
    - load:
        dataset: TCGA
    - predict:
        target: survives_1_yr

Here, we see the YAML file begin with 3 dashes - as is customary in the YAML file format
and then a dictionary begins, with a key denoting the name of the job 'Example_1_predict_survival'.
The value/item under this dictionary key is a list of key:value pairs, denoting workflow functions
being called. The first item is the load key, with a dictionary of arguments and their values
for the load function given as values for the key.

The second item in the 'Example_1_predict_survival' task is a dictionary with the key 'predict', denoting
that Phenonaut will be calling its prediction capabilities, and under this as values is another key:value
pair denoting the target column/feature within the dataset.

This page documents all workflow enabled functions such as 'load' and 'predict' as given above.

The publication SI material contains a worked example using workflow mode which can also be found here:
:doc:`tcga_example`.

Documentation on the Python API is useful for understanding the implementation details of 
:doc:`workflow mode<phenonaut>`.

I/O
***

load
----

    Load a dataset in the format denoted by a given file extension, or a PakcagedDataset known to Phenonaut. 
    There are 2 main routes to loading a dataset in workflow mode.

    1. Loading a packaged dataset (using the dataset key)
    2. Loading a user supplied file (using the file key) - normally a CSV file, but can be any file format given below, with filetype inferred through the filename extension.

    *dataset*
        The name of the packaged dataset known to Phenonaut that should be loaded. The currently available datasets are listed below:

        .. csv-table:: Available packaged datasets (case insensitive)
            :header: "Name", "Description"

            "TCGA", "The Cancer Genome Atlas (Weinstein 2013) prepared using methods described by Lee (Lee 2021)."
            "CMAP", "Connectivity Map (Lamb 2006)."
            "iris", "The Iris dataset (Yann)."
            "BreastCancer", "The Breast cancer Wisconsin (diagnostic) dataset (Wisconsin)."

        

        References:
            * Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.
            * Lee, Changhee, and Mihaela van der Schaar. "A variational information bottleneck approach to multi-omics data integration." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.
            * Lamb, Justin, et al. "The Connectivity Map: using gene-expression signatures to connect small molecules, genes, and disease." science 313.5795 (2006): 1929-1935.
            * Yann, http://yann.lecun.com/exdb/mnist/, (visited 27/4/2022)
            * Wisconsin, https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic), (visited 27/4/2022) *

        .. code-block:: yaml

            ---
            Example dataset load usage:
            - load:
                dataset: TCGA


    *file*
        The path to the file should be supplied using the *file* key. This is not required if loading a packaged dataset known to
        Phenonaut which is loaded using the dataset key described above

        .. csv-table:: Workflow file formats (* can be <ext>.gz indicating gzip compression is applied)
            :header: "Extension", "Description"
            
            "csv*",               "Standard CSV (Comma separated values) file"
            "tsv*",               "Tab separated values file"
            "h5",               "H5 data.  A 'key' argument must also be passed in conjunction with this file format,
            
            denoting the key within the H5 file that the (normally, pd.DataFrame resides under)"
        
        Filetype is inferred from the file name extension. Note that .csv and .tsv files may be gzipped (.csv.gz and .tsv.gz).

    *metadata*
        In addition to specifying the file location, a *metadata* dictionary may also be given, allowing special keywords for the reading of data in different formats.
        Special keys for the metadata dictionary are listed below (See Pandas documentation for read_csv for more in-depth information):

        - 'sep': define separator for fields within the file (by default ',', but if none is supplied and a TAB character is found in the first line of the file, then it is assumed sep should be a tab character)
        - 'skiprows': define a number of rows to skip at the beginning of a file
        - 'header_row_number' : may be a single number, or list of numbers denoting header rows.
        - 'transpose' : In the case of some transcriptomics raw data, transposition is required to have samples row-wise. Therefore the table must be transposed. Set to True to transpose.
        - 'index_col' : Column to use as the row labels of the CSV given as either the column names or as numerical indexes. Note: index_col=False can be used to force pandas to not use the first column as the index, e.g. when you have a malformed file with delimiters at the end of each line.
        - 'key' : If an h5 file is specified, then this is the key under which the pandas dataframe representing the dataset resides.
        
        .. code-block:: yaml
            
            ---
            Example load usage:
            - load:
                file: cell_paint.csv
                metadata:
                    sep: ','
                    transpose: false
                    features_prefix:
                    - feat_


write_csv
---------

    Writes a dataset to a CSV file. As well as the keys defined below, additional keys may be supplied
    which are passed to Pandas.DataFrame.to_csv as kwargs, allowing fully flexible output.
    
    See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv for a full listing of additional arguments.


    *target_dataset*
        Index or name of dataset which should be writen to the file. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *file*
        The path to the file where output should be writen.

    .. code-block:: yaml
        
        ---
        Example write_csv usage:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        -write_csv:
            file: pca_cell_paint.csv

write_multiple_csvs
-------------------

    
    Often it is useful to write a CSV file per plate within the dataset, or group the data by some other
    identifier. This write_multiple_csvs workflow function provides this functionality.

    Additional keys may be supplied which are passed to Pandas.DataFrame.to_csv as kwargs, allowing fully
    flexible output. See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv
    for a full listing of additional arguments.

    *split_by_column*
        the name of the column to be split on
    *output_dir*
        the target output directory
    *file_prefix*
        optional prefix for each file.
    file suffix: str
        optional suffix for each file.
    file_extension: str
        optional file extension, by default '.csv'

    .. code-block:: yaml
        
        ---
        Example write_multiple_csvs usage:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        -write_multiple_csvs:
            split_by_column: plate_id
            output_dir: split_output
            file_prefix: split

Transforms
**********

add_well_id
-----------

    Often, we would like to use well and column numbers to resolve a more traditional alpha-numeric WellID notation, such as A1, A2, etc. This can be achieved through calling this workflow function.

    If a dataset contains numerical row and column names, then they may be translated into standard letter-number well IDs. The arguments dictionary may contain the following keys, with their values denoted as bellow:

    *numerical_column_name*
        Name of column containing numeric column number, if not supplied, then behaves as if “COLUMN”.

    *numerical_row_name*
        Name of column containing numeric column number, if not supplied, then behaves as if “ROW”.

    *plate_type*
        Plate type - note, at present, only 384 well plate format is supported, if not supplied, then behaves as if 384.

    *new_well_column_name*
        Name of new column containing letter-number well ID, if not supplied, then behaves as if default “Well”.

    *add_empty_wells*
        Should all wells from a plate be inserted, even when missing from the data, if not supplied, then behaves as if False.

    *plate_barcode_column*
        Multiple plates may be in a dataset, this column contains their unique ID, if not supplied, then bahaves as if None.

    *no_sort*
        Do not resort the dataset by well ID, if not supplied, then behaves as if False

    .. code-block:: yaml
        
        ---
        Example add_well_id:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - add_well_id:
            numerical_column_name: COLUMN_X
            numerical_row_name: ROW_Y
            plate_type: 384
            add_empty_wells: true


copy_column
-----------

    Copys the values of one column within a dataset to another.
    
    Supply either 'to' and 'from' keys, or alternatively, simply from:to key-value
    pairs denoting how to perform the copy operation.

    Argument details
    
    .. code-block:: yaml
        
        ---
        Example copying a column:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - copy_column:
            - from: feat_1
            - to: new_feat_1
        - copy_column:
            - feat_2: new_feat_2

rename_column and rename_columns
--------------------------------
    Renames a single, or multiple columns. The arguments dictionary should contain key:value pairs, where the key is the old column name and the value is the new column name.

    Supply either 'name_from' and 'name_to' keys, or alternatively, simply from:to key-value pairs denoting how to perform the rename operation. Multiple columns can be renamed in a single call, using multiple dictionary entries.
        
    .. code-block:: yaml
        
        ---
        Example rename columns:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - rename_column:
            - feat_1: new_feat_1
            - feat_2: new_feat_2
            - name_from: feat_3
            - name_to: new_feat_3

Filtering
*********

filter_columns
--------------
    Datasets may have columns defined for keeping or removal. This function also provides a convenient way to reorder dataframe columns.

    *keep*
        Only matching columns are kept if true. If false, they are removed. By default, True.
    *column_names*
        Can be a string, or list of strings to match (or regular expressions). Can be used interchangably with column_name.
    *column_name*
        Singular column to keep (or regular expression to match).  Can be used interchangably with column_names.
    *regex*
        If true, perform regular expression matching. By default false.
    
    .. code-block:: yaml
        
        ---
        Example filtering coiumns:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - filter_columns:
            column_name: feat_

    The above example removes all metadata, keeping only feature (or columns beginning with feat\_).

filter_rows
--------------
    Alows keeping of only rows with a certain value in a certain column. Takes as arguments a dictionary
    containing query_column key:value pair and one of query_value, query_values or values key:value pairs.

    *query_column*
        Name of the column that should match the value below
    *query_value*
        Value to match. Note: query_value, query_values and values can be used interchangably.
    *query_values*
        values to match (as a list). Note: query_value, query_values and values can be used interchangably.
    *values*
        values to match (as a list.) Note: query_value, query_values and values can be used interchangably.
    *keep*
        If True then rows matching the query are kept, if False, then rows matching are discarded, and non-matching rows kept.

    .. code-block:: yaml
        
        ---
        Example filtering rows:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - filter_rows:
            query_column: valid_well
            query_value:
                - 1
                - true
                - yes

    The above example only keeps rows where the value in the valid_well column is 1, true, or yes.

filter_correlated_features
--------------------------

    Perform filter of highly correlated features (as calculated by Pearson correlation coefficient) by either by removal
    of features correlated above a given theshold, or uses the iterative removal of features with the highest R^2 against
    another feature. The arguments dictionary should contain a threshold or n key:value pair, not both. A key of threshold
    and float value defines the correlation above which, features should be removed. If the n key is present, then features
    are iteratively removeduntil n features remain.

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *threshold*
        If this key is present, then it activates threshold mode, where by calculated correlations, above which should be removed. A good value for this threshold is 0.9.
    *n*
        If this key is present, then the number of features to keep is defined this way. The process works through iteratively removing features ordered by the most correlated
        until the number of features is equal to n. If threshold is also present, then n acts as a minimum number of features and feature removal will stop,
        no matter the correlations present in the dataset.
    *drop_columns*
        If drop columns is True, then not only will features be removed from the dataset features list, but the columns for these features will
        be removed from the dataframe. If absent, then the behaviour is as if False was supplied as a value to this key:value pair.

    .. code-block:: yaml
        
        ---
        Example filtering correlated features:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - filter_correlated_features:
            n: 10

VIF_filter_features
-------------------

    Performs variance inflation factor (VIF) filtering on a dataset, removing features which are not detrimental to capturing variance. More information available: https://en.wikipedia.org/wiki/Variance_inflation_factor

    This can be a computationally expensive process as the number of linear regressions required to be run is almost N^2 with features.

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *vif_cutoff*
        The VIF cutoff value, above which features are removed. Features with VIF scores above 5.0 are considered highly correlated. If not supplied, then behaviour is as if a value of 5.0 was supplied.
    *min_features*
        Removal of too many features can be detrimental. Setting this value sets a lower limit on the number of features which must remain. If absent, then behaviour is as if a value of 2 was given.
    *drop_columns*
        If drop columns is True, then not only will features be removed from the dataset features list, but the columns for these features will be removed from the dataframe. If absent, then behaviour is as if False was supplied.

    .. code-block:: yaml
        
        ---
        Example VIF filtering:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        -VIF_filter_features:
            min_features: 3


filter_correlated_and_VIF_features
----------------------------------

    Ideally, Variance Infation Factor (VIF) filtering would be applied to very large datasets. Due to the almost n^2
    number of linear regression required as features increase, this is not possible on datasets with a large
    number of features - such as methylation datasets. We therefore must use other methods to reduce the features
    to a comfortable level allowing VIF to be performed. This class calculates correlations between all features
    and iteratively (Pearson correlation coefficient), removing features with the highest R^2 against another feature.
    Once the number of featurs is reduced to a level suitable for VIF, VIF is performed.

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *n_before_vif*
        Number of features to remove before applying VIF. This is required when dealing with large datasets which would be too time consuming to process entirely with VIF. Features are removed iteratively, selecting the most correlated features and removing them. If this key:value pair is absent, then it is as if the value of 1000 has been supplied.
    *vif_cutoff*
        The VIF cutoff value, above which features are removed. Features with VIF scores above 5.0 are considered highly correlated. If not supplied, then behaviour is as if a value of 5.0 was supplied.
    *drop_columns*
        If drop columns is True, then not only will features be removed from the dataset features list, but the columns for these features will be removed from the dataframe. If absent, then behaviour is as if False was supplied.

    .. code-block:: yaml
        
        ---
        Example filtering correlated features and then VIF filtering:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        -filter_correlated_and_VIF_features:
            n_before_vif: 10


if_blank_also_blank
-------------------
    Often it is required to clean or remove rows not needed for inclusion into further established pipelines/ workflows. This workflow function allows the ability to remove values from a column on the condition that onther column is empty.
    
    *query_column*
        Value is the name of the column to perform the query on.
    *regex_query*
        Value is a boolean denoting if the query column value should be matched using a regular expression. If omitted, then behaves as if present and False.
    *target_column*
        Value is a string, denoting the name of the column which should be blanked.
    *target_columns*
        Value is a list of strings, denoting the names of columns which should be blanked.
    *regex_targets*
        Value is a boolean denoting if the target column or multiple target columns defined in target_columns should be matched using a regular expression. If absent, then behaves as if False was supplied.


    .. code-block:: yaml
        
        ---
        Example conditionally removing columns:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - if_blank_also_blank:
            query_column: dispense_volume
            target_column: valid_well


Dimensionality reduction
************************

pca
---
    Perform PCA dimensionality reduction technique. If no arguments are given then 2D PCA is applied to the dataset with
    the highest index which is usually the last inserted dataset.

    *target_dataset*
        Index or name of dataset which should have the dimensionality reduction applied. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *ndims*
        Number of dimensions to which the PCA should reduce the features. If absent, then defaults to 2.
    *center_on_perturbation_id*
        PCA should be recentered on the perturbation with ID. If absent, then defaults to None, and no centering is performed.
    *center_by_median*
        If true, then median of center_on_perturbation is used, if False, then the mean is used.
    *fit_perturbation*
        PCA may be fit to only the included IDs, before the transform is applied to the whole dataset.

    .. code-block:: yaml
        
        ---
        Example PCA:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2

tnse
----
    Perform the t-SNE dimensionality reduction technique. If no arguments are given then 2D t-SNE is applied to the dataset with
    the highest index which is usually the last inserted dataset.

    *target_dataset*
        Index or name of dataset which should have the dimensionality reduction applied. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *ndims*
        Number of dimensions to which the t-SNE should reduce the features. If absent, then defaults to 2.
    *center_on_perturbation_id*
        t-SNE should be recentered on the perturbation with ID. If absent, then defaults to None, and no centering is performed.
    *center_by_median*
        If true, then median of center_on_perturbation is used, if False, then the mean is used.

    .. code-block:: yaml
        
        ---
        Example t-SNE:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - tsne:
            ndims: 2

umap
----
    Perform the UMAP dimensionality reduction technique. If no arguments are given then 2D UMAP is applied to the dataset with
    the highest index which is usually the last inserted dataset.

    *target_dataset*
        Index or name of dataset which should have the dimensionality reduction applied. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *ndims*
        Number of dimensions to which the UMAP should reduce the features. If absent, then defaults to 2.
    *center_on_perturbation_id*
        UMAP should be recentered on the perturbation with ID. If absent, then defaults to None, and no centering is performed.
    *center_by_median*
        If true, then median of center_on_perturbation is used, if False, then the mean is used.

    .. code-block:: yaml
        
        ---
        Example UMAP:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - umap:
            ndims: 2

Measurements
************

scalar projection
-----------------
    Add a column for the scalar projection. Calculates the scalar projection and scalar rejection, quantifying on and off
    target phenotypes, as used in: Heiser, Katie, et al. “Identification of potential treatments for COVID-19 through
    artificial intelligence-enabled phenomic analysis of human cells infected with SARS-CoV-2.” BioRxiv (2020).

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    
    *target_perturbation_column_name*
        Column name to match the target_perturbation_column value defined below, usually a column capturing control names.

    *target_perturbation_column_value*
        value to be found in the column defined in target_perturbation_column_name
    
    *output_column_label*
        Output from the scalar projection will have the form:
            on_target\_<output_column_label>

            off_target\_<output_column_label>

        if this is missing, then it is set to target_perturbation_column_value
        
    .. code-block:: yaml
        
        ---
        Example scalar projection:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - scalar_projection:
            target_perturbation_column_name: control_name
            target_perturbation_column_value: positive_control
            output_column_label: sproj


euclidean_distance
------------------
    Add a column for the euclidean distance between all perturbations and a target perturbation.

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    
    *target_perturbation_column_name*
        Column name to match the target_perturbation_column value defined below, usually a column capturing control names.

    *target_perturbation_column_value*
        value to be found in the column defined in target_perturbation_column_name
    
    *output_column_label*
        Output column name for the distance measurements. If this is missing, then it is set to target_perturbation_column_value
        
    .. code-block:: yaml
        
        ---
        Example scalar projection:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - euclidean_distance:
            target_perturbation_column_name: control_name
            target_perturbation_column_value: positive_control
            output_column_label: dist


manhattan_distance
------------------
    Add a column for the manhattan distance between all perturbations and a target perturbation (also known as the cityblock distance).

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    
    *target_perturbation_column_name*
        Column name to match the target_perturbation_column value defined below, usually a column capturing control names.

    *target_perturbation_column_value*
        value to be found in the column defined in target_perturbation_column_name
    
    *output_column_label*
        Output column name for the distance measurements. If this is missing, then it is set to target_perturbation_column_value
        
    .. code-block:: yaml
        
        ---
        Example scalar projection:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - manhattan_distance:
            target_perturbation_column_name: control_name
            target_perturbation_column_value: positive_control
            output_column_label: dist

cityblock_distance
------------------
    Add a column for the cityblock distance between all perturbations and a target perturbation (also known as the manhattan distance).

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    
    *target_perturbation_column_name*
        Column name to match the target_perturbation_column value defined below, usually a column capturing control names.

    *target_perturbation_column_value*
        value to be found in the column defined in target_perturbation_column_name
    
    *output_column_label*
        Output column name for the distance measurements. If this is missing, then it is set to target_perturbation_column_value
        
    .. code-block:: yaml
        
        ---
        Example scalar projection:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - cityblock_distance:
            target_perturbation_column_name: control_name
            target_perturbation_column_value: positive_control
            output_column_label: dist

mahalanobis_distance
--------------------
    Add a column for the mahalanobis distance between all perturbations and a target perturbations.

    *target_dataset*
        Index or name of dataset which should have features filtered. If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    
    *target_perturbation_column_name*
        Column name to match the target_perturbation_column value defined below, usually a column capturing control names.

    *target_perturbation_column_value*
        value to be found in the column defined in target_perturbation_column_name
    
    *output_column_label*
        Output column name for the distance measurements. If this is missing, then it is set to target_perturbation_column_value
        
    .. code-block:: yaml
        
        ---
        Example scalar projection:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - mahalanobis_distance:
            target_perturbation_column_name: control_name
            target_perturbation_column_value: positive_control
            output_column_label: dist

Plotting
********

set_perturbation_column
-----------------------
    Within a dataset, a special flag on a column may be set indicating that this is the
    perturbation column. This is useful in plotting whereby similar perturbations/repeats
    should be the same colour etc.

    *target_dataset*
        Index or name of dataset within which we wish to set the perturbation column.
        If absent, then behaviour is as if -1 is supplied, indicating the last added dataset.
    *column*
        column name to mark as containing perturbations ids.
        
    .. code-block:: yaml
        
        ---
        Example set_perturbation_column:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - set_perturbation_column:
            column: compound_id

scatter
-------
    produce a scatter plot from a dataset. Used in conjunction with set_perturbation_column,
    similar pertubations will be grouped together.

    *target_dataset*
        Index or name of dataset which should have features plotted. If absent, then behaviour is as if -1 is supplied,
        indicating the last added dataset.
    *figsize*
        A tuple denoting the target output size in inches (!), if
        absent, then the default of (8,6) is used.
    *title*
        Title for the plot, if absent, then the default "2D scatter"
        is used.
    *peturbations*
        Can be a list of peturbations - as denoted in the
        perturbations column of the dataframe to include in the plot.
        If absent, then all perturbations are included.
    *destination*
        Output location for the PNG - required field.  An error will be thrown if omitted.
    
    .. code-block:: yaml
        
        ---
        Example scatter:
        - load:
            file: cell_paint.csv
            metadata:
                sep: ','
                transpose: false
                features_prefix:
                - feat_
        - pca:
            ndims: 2
        - set_perturbation_column:
            column: compound_id
        - scatter:
            destination: scatter_pca_cell_paint.png

Prediction
**********

predict
-------
    Profile predictors in their ability to predict a given target.
    
    Phenonaut provides functionality to profile the performance of multiple predictors against multiple views of data.
    This is exemplified in the TCGA example used in the Phenonaut paper - see
    :ref:`tcga_example` for a full walkthrough of applying this functionality to The Cancer Genome Atlas. With a given 'target'
    for prediction which is in the dataset, predict selects all appropriate predictors (classifiers for classification, regressors
    for regression and multiregressors for multi regression/view targets).  Then, enumerating all views of the data and all predictors,
    hyperparameter optimisation coupled with 5-fold cross validation using Optuna is employed, before finally testing the best hyperparameter
    sets with retained test sets. This process is automatic and requires the data, and a prediction target.  Output from this process is
    extensive and it may take a long time to complete, depending on the characteristics of your input data.  Writen output from the profiling
    process consists of performance heatmaps highlighting best view/predictor combinations in bold, boxplots for each view combination and
    a PPTX presentation file allowing easy sharing of data, along with machine readable CSV and JSON results.


    For each unique view combination and predictor, perform the following:
    
    - Merge views and remove samples which do not have features across currently needed views.
    - Shuffle the samples.
    - Withhold 20% of the data as a test set, to be tested against the trained and hyperparameter optimised predictor.
    - Split the data using 5-fold cross validation into train and validation sets.
    - For each fold, perform Optuna hyperparameter optimisation for the given predictor using the train sets, using hyperparameters described by the default predictors for classification, regression and multiregression.

    *output_directory*
        Directory into which profiling output (boxplots, heatmaps, CSV, JSON
        and PPTX should be written).
    *dataset_combinations*
        If multiple datasets are already loaded, then lists of
        'views' may be specified for exploration. If None, or this argument is absent, then all combinations
        of available views/Datasets are enumerated and used.
    *target*
        The prediction target, denoted by a column name given here which exists in loaded datasets.
    *n_splits*
        Number of splits to use in the N-fold cross validation, if absent, then the default of 5 is used.
    *n_optuna_trials*
        Number of Optuna trials for hyperparameter optimisation, by default 20. This drastically impacts
        runtime, so if things are taking too long, you may wish to lower this number.  For a more thorough
        exploration of hyperparameter space, increase this number.
    *optuna_merge_folds*
        By default, each fold has hyperparameters optimised and the trained
        predictor with parameters reported.  If this optuna_merge_folds is true,
        then each fold is trained on and and hyperparameters optimised across
        folds (not per-fold). Setting this to False may be useful depending on
        the intended use of the predictor. It is believed that when False, and
        parameters are not optimised across folds, then more accurate prediction
        variance/accuracy estimates are produced. If absent, behaves as if false.
    *test_set_fraction*
        When optimising a predictor, by default a fraction of the total data is
        held back for testing, separate from the train-validation splits. This
        test_set_fraction controls the size of this split. If absent, then the
        default value of 0.2 is used.
    
    .. code-block:: yaml
        
        ---
        Example_1_predict_survival:
        - load:
            dataset: TCGA
        - predict:
            target: survives_1_yr

