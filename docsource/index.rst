.. Phenonaut documentation master file, created by
   sphinx-quickstart on Tue May  3 17:00:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: phenonaut.png
   :align: right

Welcome to the Phenonaut documentation!
=======================================


Phenonaut is a framework for applying workflows to multi-omics data. Originally
targeting high-content imaging and the exploration of phenotypic space, with
different visualisations and metrics, Phenonaut allows now operates in a data
agnostic manner, allowing users to describe their data (potentially
multi-view/multi-omics) and apply a series of generic or specialised
data-centric transforms and measures.



Phenonaut operates in 2 modes:

   #. As a Python package, importable and callable within custom scripts.
   #. Operating on a workflow defined in either YAML, or JSON, allowing integration of complex chains of Phenonaut instructions to be integrated into existing workflows and pipelines. When built as a package and installed, workflows can be executed with:

      .. code-block:: bash
         
         python -m phenonaut workflow.yml

User guide
==========
Alongside the API documentation a crash-course userguide is available here:
:doc:`userguide`.

A breakdown and guide to workflow mode and commands can be found here:
:doc:`workflow_guide`.




.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   API documentation <phenonaut.rst>
   userguide
   publication_examples
   workflow_guide


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
