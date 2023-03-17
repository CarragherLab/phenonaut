# Changelog

All notable changes to this project will be documented in this file under headings Added, Changed, and Fixed

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

