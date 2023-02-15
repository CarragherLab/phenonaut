rm -f phenonaut.rst phenonaut.data.rst phenonaut.metrics.rst phenonaut.output.rst phenonaut.packaged_datasets.rst phenonaut.predict.rst phenonaut.transforms.rst modules.rst phenonaut.predict.default_predictors.rst phenonaut.predict.default_predictors.pytorch_models.rst
sphinx-apidoc -o . .. ../example*.py
rm modules.rst
make html
