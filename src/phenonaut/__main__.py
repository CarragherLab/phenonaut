# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

"""
Phenonaut workflows are an alternative way of running phenonaut transforms allowing
the processing of data as defined in YAML or JSON workflow files.
"""
import argparse
from pathlib import Path

from phenonaut.workflow import Workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Phenonaut workflow")
    parser.add_argument(
        "workflow_path", help="YAML or JSON file containing Phenonaut workflow"
    )
    args = parser.parse_args()

    workflow_path = Path(args.workflow_path)

    ph0_workflow = Workflow(workflow_path)
