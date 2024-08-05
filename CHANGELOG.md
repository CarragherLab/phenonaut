# Changelog

All notable changes to this project will be documented in this file under headings Added, Changed, and Fixed

## [2.0.5] - 2024-06-26

### Added
- Added filter_on_identifiers method to phenonaut.data.Datasets, allowing easy filtering of train/val/test sets
- Added filter_datasets_on_identifiers method to phenonaut objects to filter all datasets
- Added ContrastiveEncoder
- Added the shrink method to phenonaut objects and datasets
- Added calculation of type2 errors for permutation testing distinctness
- Added get_unique_treatments, and num_features properties of datasets
- Added ability to iterate Phenonaut objects, returning their datasets
- Added ability to call len on Phenonaut objects, returning the number of datasets they contain

### Changed
- Refactored phenonaut.metrics
- Applied ruff checks/fixes


## [2.0.3] - 2024-03-26

### Fixed
- Exclude pptx files from black formatting
- Dataset constructor raises an error if features passed as metadata are not of type list
- merged dataset deletion with passed datasets
- test using dataset_groupby, corrected to groupby_datasets

## [2.0.2] - 2024-03-06

### Fixed
- RandomForest regressor no longer uses auto max_features hyperparameter, making it compatible with scikit-learn 1.1 onwards.

## [2.0.1] - 2024-03-01

### Changed
- The way versioning works internally to Phenonaut.
- Updated pyproject.toml to include powerpoint templates

## [1.5.1] - 2023-10-19

### Added
- New class of Error NotEnoughRowsError added to better flag runtime errors.
- Added checks to mp_value_score which ensure grouped dataframe groups have at least 3 rows required for calculations.
- In mp_value_score, groups with < 3 rows may be ignored by calling the function with raise_error_for_low_count_groups = False, in which case np.nan values will be returned for the group.

### Changed
- Improved tests for mp_value_score, checking for correct behaviour within small groups with <3 rows.


## [1.5.0] - 2023-09-27

### Changed
- Refactored package to use pyproject.toml for install/build
- Updated classifier hyperparameters, `max_features='auto'` has been deprecated in scikit 1.1 and will be removed in 1.3. Now explicitly set to `max_features='sqrt'`.
- Removed progressbar2 in favour of tqdm throughout.

### Fixed
- predict.profile updated to work with newer pandas
- Phenonaut.merge_datasets now honours the remove_merged flag


## [1.4.3] - 2023-08-16
### Added
- random_state argument for percent replicating, allowing passing of a np.random.Generator, or an int to seed a new Generator


## [1.4.1] - 2023-07-28
### Fixed
- bug in Phenonaut.merge_datasets
- bug in data.Datasets.groupby

## [1.4.0] - 2023-07-18
### Added
- merge_datasets method added to Phenonaut objects
- mp_value_score metric added to phenonaut.metrics.performance (doi:10.1177/1087057112469257)
- added __repr__ to Phenonaut objects


## [1.3.8] - 2023-06-22
### Changed
- Removed py3.10 style Unions, favouing the old style typing.Union

## [1.3.7] - 2023-06-22
### Fixed
- bug in the generation of Scree plots from fitted PCA transformers 


## [1.3.6] - 2023-06-01
### Added
- groupby function to dataset, allowing splitting on one dataset into many

## [1.3.5] - 2023-03-29
### Added
- orient argument to write_boxplot_to_file allowing horizontal or vertical plotting.

## [1.3.4] - 2023-03-27

### Added
- Ability to supply features argument to new Datasets rather than wrapping in metadata dictionary
- data property to Phenonaut objects to return phe[-1].df[phe[-1].features].values
- weights argument to KNNImputer

### Changed
- Percentile argument for percent_replicating and percent_compact is no longer required, and inferred to be 95th (if higher is better), or 5th if lower is better

### Fixed
- Broken test caused by tempfile creation and phe.revert

## [1.3.3] - 2023-03-24

### Added
- Added imputers (SimpleImputer, KNNImputer, RFImputer) to transforms.imputers

## [1.3.2] - 2023-03-24

### Added
- get_dataset_names to phenonaut.Phenonaut objects

### Fixed
- PackagedDataset loaders do not keep a H5store open, allowing multiple concurrent runs in HPC environment.

### Changed
- percentile_cutoff now optional for percent_replicating and percent_compact, allowing inference if None for 95th/5th percentile depending on similarity metric (higher or lower better)

## [1.2.8] - 2023-03-16

### Added
- PackagedDataset for BROAD-MOA assignment, derived from the LINCS Cell Painting project

## [1.2.7] - 2023-03-16

### Added
- subtract_func_results_on_features to phenonaut.data.Dataset, which allows subtraction of func results to samples (reworked subtract_median and subtract_mean to call this generic function)

### Fixed
- Percent replicating docstring


## [1.2.6] - 2023-03-15

### Added
- PackagedDataset for JUMP-MOA compound set, allowing easy access to this metadata
- added percent_compact performance measure
- functionality to metrics.utils for manipulating percent compact results into plots
- added percent_replicating_results_dataframe_to_95pct_confidence_interval to phenonaut.metrics.utils, allowing the calculation of confidence intervals on percent replicating scores by resampling the null distribution.
- added restrict_evaluation_query to phenonaut.metrics.performance.percent_replicating to allow restriction of evaluation to certain rows without removing unused compounds from the pool of non-matching compounds comprising the null distribution.

### Changed
- Renamed phenonaut.metrics.utils.percent_replicating_results_dataframe_to_percentile_vs_replicating to percent_replicating_results_dataframe_to_percentile_vs_percent_replicating and changed returned fraction replicating to percent replicating (X100) to keep terminology consistent.
- Changed the above newly renamed percent_replicating_results_dataframe_to_percentile_vs_percent_replicating return_fraction (by default True) to return_counts (by default False). If True, then y values denote the counts of replicates which were deemed replicating. If False, then percent replicating is returned.
- Renamed phenotypic metrics with magic values in phenonaut.metrics.non_ds_phenotypic_metrics from metric_with_magic_values to metrics_with_magic_values


## [1.2.5] - 2023-03-01

### Added
- added NaN handling capability within phenonaut dataset class
- added Tests for NaN handling functionality


## [1.2.4] - 2023-03-01

### Added
- revert method to Phenonaut object, enabling quick reload of already saved Phenonaut objects.
- arguments and capabilities to percent_replicating, enabling: automatic naming of performance files from parameters, saving run parameters to a JSON file.
- added new percent_replicating_summarise_results function to metrics.utils, allowing interrogation of files or directories containing percent_replicating results.
- phenonaut.metrics.non_ds_phenotypic_metrics containing non phenonaut Dataset aware generic similarity/distance methods.

### Fixed
- fixed bug in Phenonaut.save, whereby overwrite_existing had no effect, and existing files were always overwritten.


## [1.2.3] - 2023-02-23

### Added
- metrics.utils with function for manipulating percent replicating results into plots
- phenonaut.metrics.performance.percent_replicating can now operate with other similarity and distance functions.

### Fixed
- phenonaut.metrics.performance.percent_replicating now uses Spearman's rank as per the Cell paper (Way, Gregory P., et al. "Morphology and gene expression profiling provide complementary information for mapping cell state." Cell systems 13.11 (2022): 911-923).


## [1.2.2] - 2023-02-21

### Fixed
- fixed issue concerning continuing to work on Phenonaut objects after saving
- percent replicating

### Changed
- remove_blocklist_features now considers prefixed columns and allows removal of features and non-feature columns.
- moved percent replicating to a new submodule: phenonaut.metrics.performance

## Added
- get_df_features_perturbation_column function to Phenonaut and Dataset objects allowing clean interface for code to get at underlying DataFrames regardless if passed a Phenonaut object or Dataset.


## [1.2.1] - 2023-02-16

### Fixed
- fixed missing install requirements (torch, torchvision, torchaudio)


## [1.2.0] - 2023-02-15

### Added
- added requests, optuna, and mvlearn libraries to the setup.cfg install_requires list
- added pytorch to the build requirements in pyproject.toml
- added a conftest file to keep pytest fixtures
- added PackagedDataset for CMAP level 4 data

### Changed
- example1_TCGA now constructs TCGA PackagedDataset with download=True, always downloading TCGA if absent
- removed redundant pytest fixture-only files
- maximum python version back to 3.10 because of pytorch compatibility
- removed deprecated imports which broke some tests
- removed unused keep_archive argument for packaged_datasets.base._download
### Fixed
- fixed linting errors in test suite
- fixed dataset test errors
- fixed ModuleNotFoundError for torch by adding it to build dependency
- fixed TSNE pytest error
- fixed default arguments to packaged_datasets.base._download to be consistent with docstring


## [1.1.6] - 2023-02-08

### Added
- This changelog noting all recent changes going forward

### Changed
- Removed ambiguity between GenericTransform and Transform base classes by removing GenericTransform. Now objects which are callable or have fit/transform/fit_transform may inherit from the transform class and be eaily used in Phenonaut. Below we see the easy way scikit-learn's PCA can be wrapped:

        transformer=Transformer(PCA, constructor_kwargs={'n_components':2})
        t_pca.fit(phe.ds, groupby="BARCODE")
        t_pca.transform(phe.ds)
- Cleaned up tests
- RobustMAD now moved to inherit from Transformer, allowing the groupby argument, and for normalisations to happen on a per-plate basis
