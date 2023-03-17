# Phenonaut

A toolkit for multi-omic phenotypic space exploration. 


## Description
<img style="float: right;" src="phenonaut.png">

Phenonaut is a framework for applying workflows to multi-omics data. Originally targeting high-content imaging and the exploration of phenotypic space, with different visualisations and metrics, Phenonaut allows now operates in a data agnostic manner, allowing users to describe their data (potentially multi-view/multi-omics) and apply a series of generic or specialised data-centric transforms and measures.  

Phenonaut operates in 2 modes:

- As a Python package, importable and callable within custom scripts.
- Operating on a workflow defined in either YAML, or JSON, allowing integration of complex chains of Phenonaut instructions to be integrated into existing workflows and pipelines. When built as a package and installed, workflows can be executed with:
```python -m phenonaut workflow.yml``` .


## Structrure
Datasets are read into the dataset class, aided by a yaml file describing the underlying data (see config/ for example yaml data definition files). Pandas dataframes are created representing the data (a Phenonaut object may hold multiple dataset objects), along with two additional pieces of data. 
1) A features list, accessible with .features property of a dataframe. Initially defined by the data definition workflow.
2) perturbation_column, optional column which gives a unique ID to the treatment performed on the well/vial/data.
3) Metadata, optional dictionary containing metadata for the dataset.

## Documentation

[Start here](https://carragherlab.github.io/phenonaut/)
- [User guide](https://carragherlab.github.io/phenonaut/userguide.html)
- [API documentation](https://carragherlab.github.io/phenonaut/phenonaut.html)
- [Publication examples](https://carragherlab.github.io/phenonaut/publication_examples.html)
- [Workflow mode](https://carragherlab.github.io/phenonaut/workflow_guide.html)



Copyright © The University of Edinburgh, 2023.

Development has been supported by GSK.
