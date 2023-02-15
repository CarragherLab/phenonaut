# Changelog

All notable changes to this project will be documented in this file under headings Added, Changed, and Fixed

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

