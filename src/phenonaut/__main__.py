# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

"""
Phenonaut workflows are an alternative way of running phenonaut transforms allowing
the processing of data as defined in YAML or JSON workflow files.
"""
from pathlib import Path
import fire
import phenonaut
import phenonaut.data
import phenonaut.integration
from phenonaut.workflow import Workflow
import pandas as pd


def main():
    fire.Fire(PhenonautCLI())


class PhenonautCLI:
    def workflow(self, workflow_path: str):
        """Workflow mode

        Parameters
        ----------
        workflow_path : str
            Path to workflow YAML file
        """
        return Workflow(Path(workflow_path))

    def _should_skip(self, phe_file, phenotypic_metric, output_path, test_split_only):
        return False

    def cdu(
        self,
        phe_file: str,
        phenotypic_metric: str,
        quiet: bool = False,
        name: str | None = None,
        apply_standard_scaler: bool = True,
        apply_PCA: bool = True,
        calc_compactness: bool = True,
        calc_distinctness: bool = True,
        calc_uniqueness: bool = True,
        output_path: str = 'working_dir/results/',
        test_split_path: str | None = None,
        split_field: str | None = None,
        centering_query: str | None = None,
        results_filter_path: str | None = None,
        results_filter_fields: str | list | None = None,
    ):
        from .metrics import run_cdu_benchmarks, write_cdu_json
        from .metrics.non_ds_phenotypic_metrics import non_ds_phenotypic_metrics
        from phenonaut.transforms import PCA, StandardScaler

        if phenotypic_metric == "all":
            print("Performing CDU calculation using all metrics")
            for phenotypic_metric in non_ds_phenotypic_metrics.keys():
                if phenotypic_metric == "Connectivity":
                    continue
                print("Current metric =", phenotypic_metric)
                self.cdu(
                    phe_file,
                    phenotypic_metric,
                    quiet,
                    name,
                    apply_standard_scaler,
                    apply_PCA,
                    calc_compactness,
                    calc_distinctness,
                    calc_uniqueness,
                    output_path,
                    test_split_path,
                    split_field,
                    centering_query,
                )
            return
        print(split_field, type(split_field))

        if self._should_skip(
            phe_file, phenotypic_metric, output_path, test_split_path is not None
        ):
            print(
                f"Phe file {phe_file} already profiled with {phenotypic_metric} in {output_path}, skipping"
            )
            return

        phenotypic_metric = non_ds_phenotypic_metrics[phenotypic_metric]
        ds = phenonaut.Phenonaut.load(Path(phe_file)).ds
        if test_split_path is not None:
            if split_field is None:
                raise ValueError(
                    "Splits JSON supplied, but no split_field, please provide the column/field name to match to splits, this is likely 'moa' or 'Metadata_jump_moa' etc."
                )
            test_split_path = Path(test_split_path)
            import json

            test_split_identifiers = json.load(open(test_split_path))['test']
            if isinstance(split_field, list):
                print(ds.df)
                test_split_identifiers = pd.DataFrame.from_records(
                    test_split_identifiers, columns=('pert_id', 'pert_idose_uM')
                )
                ds.df = test_split_identifiers.merge(
                    ds.df,
                    left_on=('pert_id', 'pert_idose_uM'),
                    right_on=('pert_id', 'pert_idose_uM'),
                    how='left',
                )
            else:
                test_split_identifiers.append("DMSO")
                ds.df = ds.df.query(f"{split_field} in @test_split_identifiers")
        if pd.isna(ds.data).sum().sum() > 0:
            print("NaNs detected")
            raise ValueError(
                f"{pd.isna(ds.data).sum().sum()} NaNs found in dataset (present in features)"
            )

        if name is None:
            if isinstance(phe_file, (str, Path)):
                name = Path(phe_file).stem
            else:
                name = ds.name
        if apply_standard_scaler:
            sscaler = StandardScaler()
            sscaler.fit_transform(ds)

        if not apply_PCA:
            print("Running without PCA")
            results = run_cdu_benchmarks(
                ds=ds,
                replicate_groupby=None,
                dmso_query=None,
                run_percent_replicating=calc_compactness,
                run_permutation_test_against_dmso=calc_distinctness,
                run_auroc=calc_uniqueness,
                return_full_results=False,
                phenotypic_metric=phenotypic_metric,
                quiet=quiet,
            )
            write_cdu_json(
                results,
                output_path,
                name,
                phenotypic_metric.name,
                not test_split_path is None,
            )

        if apply_PCA:
            print("Running with PCA")

            pca = PCA(ndims=0.995)
            pca.fit_transform(ds, center_on_perturbation_id=centering_query)

            results = run_cdu_benchmarks(
                ds=ds,
                replicate_groupby=None,
                dmso_query=None,
                run_percent_replicating=calc_compactness,
                run_permutation_test_against_dmso=calc_distinctness,
                run_auroc=calc_uniqueness,
                return_full_results=False,
                phenotypic_metric=phenotypic_metric,
                quiet=quiet,
            )
            write_cdu_json(
                results,
                output_path,
                name,
                f"{phenotypic_metric.name}_PCA",
                not test_split_path is None,
            )

    def integrate(
        self,
        phe_files: list | str,
        output_dataset_path: str,
        method: str,
        splits_json: str | Path | None,
    ):
        """Integrate datasets

        Parameters
        ----------
        phe_files : list[str] | str
            List of phenonaut files, or a string representing files separated by commas.
            The easiest way to pass this on the command line is using the form as follows with no
            spaces:
            ```bash
            phenonaut integrate phe1.pkl,phe2.pkl concat integrated_output.pkl
            ```
        output_dataset_path : str
            Output path for integrated data stored as a Phenonaut object
        method : str
            Integration method, can be one of:
            - 'mvmds' - for MultiView Multi-Dimensional Scaling
            - 'splitae' - for a split AutoEncoder
            - 'concat' - for simple concatenation of features
        splits_json: str | Path
            Path to json split file


        Returns
        -------
        phenonaut.Phenonaut
            Returns a Phenonaut object containing the integrated data.

        Raises
        ------
        FileNotFoundError
            _description_
        """

        print("Integrating:", phe_files)

        if isinstance(phe_files, str):
            if "," in phe_files:
                phe_files = [Path(f.strip()) for f in phe_files.split(",")]
            else:
                phe_files = [Path(phe_files)]
        if isinstance(phe_files, list):
            phe_files = [Path(f) for f in phe_files]

        for f in phe_files:
            if not f.exists():
                raise FileNotFoundError(f"Could not find phenonaut file {f}")

        # phe_files is a list[Path] of existing phe files
        if len(phe_files) == 1:
            phe = phenonaut.Phenonaut.load(phe_files[0])
            datasets = phe.datasets
        else:
            datasets = [phenonaut.Phenonaut.load(f).ds for f in phe_files]

        integrated_ds = phenonaut.integration.integrate_datasets(
            datasets=datasets, integration_method=method
        )

        phenonaut.Phenonaut(integrated_ds).save(Path(output_dataset_path))


if __name__ == "__main__":
    fire.Fire(PhenonautCLI())
