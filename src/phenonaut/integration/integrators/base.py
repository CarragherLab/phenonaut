import phenonaut
import pandas as pd
import numpy as np

import phenonaut.data

__all__ = ["ViewIntegrator"]


class ViewIntegrator:
    def __init__(
        self, name: str, has_fit: bool, has_transform: bool, has_fit_transform: bool
    ) -> None:
        self.name = name
        self.has_fit = has_fit
        self.has_transform = has_transform
        self.has_fit_transform = has_fit_transform

    @staticmethod
    def _phe_lists_to_df_lists(
        phe_ds_lists: (
            list[phenonaut.data.Dataset]
            | list[list[phenonaut.data.Dataset]]
            | list[phenonaut.Phenonaut]
            | list[list[phenonaut.Phenonaut]]
        ),
    ) -> list[phenonaut.data.Dataset] | list[list[phenonaut.data.Dataset]]:
        if isinstance(phe_ds_lists[0], list):
            for i in range(len(phe_ds_lists)):
                for j in range(len(phe_ds_lists[i])):
                    if isinstance(phe_ds_lists[i][j], phenonaut.Phenonaut):
                        phe_ds_lists[i][j] = phe_ds_lists[i][j].ds
        else:
            for j in range(len(phe_ds_lists)):
                if isinstance(phe_ds_lists[j], phenonaut.Phenonaut):
                    phe_ds_lists[j] = phe_ds_lists[j].ds
        return phe_ds_lists

    @staticmethod
    def multiplex_df(
        datasets: list[phenonaut.data.Dataset],
        how: str = "left",
        random_state: int | np.random.Generator = 7,
        drop_non_critical_data: bool = True,
        shuffle_joined_dataset: bool = True,
    ) -> tuple[pd.DataFrame, list[list[str]]]:

        if isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)

        ds_perturbation_columns = [
            (
                ds.perturbation_column
                if isinstance(ds.perturbation_column, list)
                else [ds.perturbation_column]
            )
            for ds in datasets
        ]

        if drop_non_critical_data:
            for i in range(len(datasets)):
                datasets[i].df = datasets[i].df[
                    ds_perturbation_columns[i] + datasets[i].features
                ]
                print(datasets[i].df.shape)

        feature_lists = [[f"ds1_{f}" for f in datasets[0].features]]
        df = (
            datasets[0]
            .df[ds_perturbation_columns[0] + datasets[0].features]
            .rename(columns={f: f"ds1_{f}" for f in datasets[0].features})
        )
        for i in range(1, len(datasets)):
            feature_lists.append([f"ds{i+1}_{f}" for f in datasets[i].features])
            print(i)
            df = df.merge(
                datasets[i]
                .df[ds_perturbation_columns[i] + datasets[i].features]
                .groupby(ds_perturbation_columns[i])
                .sample(1, random_state=random_state),
                left_on=ds_perturbation_columns[i - 1],
                right_on=ds_perturbation_columns[i],
                how=how,
                suffixes=("", "_y"),
            )
            df.rename(
                columns={f: f"ds{i+1}_{f}" for f in datasets[i].features}, inplace=True
            )
        for features in feature_lists:
            if np.isnan(df.loc[:, features].values).sum().sum() > 0:
                raise ValueError("NaNs found in integrated DataFrame")
        return df, feature_lists

    @staticmethod
    def multiplex_df_merge_single_samples(
        datasets: list[phenonaut.data.Dataset],
        how: str = "left",
        random_state: int | np.random.Generator = 7,
        keep_metadata_columns: bool = True,
    ) -> tuple[pd.DataFrame, list[list[str]]]:

        if isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)

        metadata_columns = [
            [
                c
                for c in ds.df.columns
                if c.startswith("Metadata_") and keep_metadata_columns
            ]
            for ds in datasets
        ]

        ds_perturbation_columns = [
            (
                ds.perturbation_column
                if isinstance(ds.perturbation_column, list)
                else [ds.perturbation_column]
            )
            for ds in datasets
        ]

        feature_lists = [[f"ds1_{f}" for f in datasets[0].features]]
        df = (
            datasets[0]
            .df[ds_perturbation_columns[0] + datasets[0].features + metadata_columns[0]]
            .rename(columns={f: f"ds1_{f}" for f in datasets[0].features})
        )
        for i in range(1, len(datasets)):
            feature_lists.append([f"ds{i+1}_{f}" for f in datasets[i].features])
            df = df.merge(
                datasets[i]
                .df[
                    tuple(
                        ds_perturbation_columns[i]
                        + datasets[i].features
                        + metadata_columns[i]
                    )
                ]
                .groupby(ds_perturbation_columns[i])
                .sample(1, random_state=random_state),
                left_on=ds_perturbation_columns[i - 1],
                right_on=ds_perturbation_columns[i],
                how=how,
                suffixes=("", "_y"),
            )
            df.rename(
                columns={f: f"ds{i+1}_{f}" for f in datasets[i].features}, inplace=True
            )
        for features in feature_lists:
            if np.isnan(df.loc[:, features].values).sum().sum() > 0:
                raise ValueError("NaNs found in integrated DataFrame")
        return df, feature_lists

    def fit_transform(
        self, datasets: list[phenonaut.data.Dataset]
    ) -> phenonaut.data.Dataset:
        if not self.has_fit_transform:
            raise NotImplementedError(f"{self.name} does not support fit_transform")
        datasets = self._phe_lists_to_df_lists(datasets)
        merged_df, merged_df_feature_lists = self.multiplex_df(datasets, "inner")
        data_for_integration = [
            merged_df.loc[:, merged_df_feature_lists[n]].values
            for n in range(len(merged_df_feature_lists))
        ]
        integrated_data = self.method.fit_transform(data_for_integration)
        del data_for_integration
        new_features = [f"ifeat_{n+1}" for n in range(integrated_data.shape[1])]
        merged_df.loc[:, new_features] = integrated_data
        new_ds = phenonaut.data.Dataset(
            self.name + " integrated", merged_df, metadata={"features": new_features}
        )
        new_ds.perturbation_column = datasets[0].perturbation_column
        return new_ds

    def transform(
        self, datasets: list[phenonaut.data.Dataset]
    ) -> phenonaut.data.Dataset:
        if not self.has_transform:
            raise NotImplementedError(f"{self.name} does not support transform")
        datasets = self._phe_lists_to_df_lists(datasets)
        merged_df, merged_df_feature_lists = self.multiplex_df(datasets, "inner")
        data_for_integration = [
            merged_df.loc[:, merged_df_feature_lists[n]].values
            for n in range(len(merged_df_feature_lists))
        ]
        integrated_data = self.method.transform(data_for_integration)
        print(integrated_data)
        del data_for_integration
        new_features = [f"ifeat_{n+1}" for n in range(integrated_data.shape[1])]
        merged_df.loc[:, new_features] = integrated_data
        new_ds = phenonaut.data.Dataset(
            self.name + " integrated", merged_df, metadata={"features": new_features}
        )
        new_ds.perturbation_column = datasets[0].perturbation_column
        return new_ds

    def fit(self, datasets: list[phenonaut.data.Dataset]) -> None:
        if not self.has_fit:
            raise NotImplementedError(f"{self.name} does not support fit")
        datasets = self._phe_lists_to_df_lists(datasets)
        merged_df, merged_df_feature_lists = self.multiplex_df(datasets, "inner")
        data_for_integration = [
            merged_df.loc[:, merged_df_feature_lists[n]].values
            for n in range(len(merged_df_feature_lists))
        ]
        _ = self.method.fit(data_for_integration)

    @staticmethod
    def merge_splits_to_single_views(
        datasets: list[list[phenonaut.data.Dataset]] | list[phenonaut.data.Dataset],
    ) -> list[phenonaut.data.Dataset]:
        """_summary_

        Parameters
        ----------
        datasets : list[list[phenonaut.data.Dataset]] | list[phenonaut.data.Dataset]
            M length list of N length lists of datasets,
            where M is of length 2 for train+test sets, and 3 for train, val, test sets
            and N is the number of views for each.

            [[View1_train, View1_val, View1_test], [[View1_train, View1_val, View1_test]]]

        Returns
        -------
        list[phenonaut.data.Dataset]
            _description_
        """
        if isinstance(datasets, phenonaut.data.Dataset):
            return datasets
        if not isinstance(datasets[0], list):
            raise ValueError(
                "Expected atasets: list[list[phenonaut.data.Dataset]] or list[phenonaut.data.Dataset], got",
                datasets,
            )
        # List of lists
        merged_views = []
        for view in datasets:
            merged_df = pd.concat([ds.df for ds in view])
            new_ds = phenonaut.data.Dataset(
                f"Merged - {';'.join([ds.name for ds in view])}",
                merged_df,
                features=view[0].features,
            )
            new_ds.perturbation_column = view[0].perturbation_column
            merged_views.append(new_ds)

        return merged_views

    def fit_train_test_transform_test(
        self,
        datasets_train: list[list[phenonaut.data.Dataset]],
        datasets_test: list[list[phenonaut.data.Dataset]],
    ) -> phenonaut.data.Dataset:
        """Fit transform for MVMDS

        Parameters
        ----------
        datasets : list[list[phenonaut.data.Dataset]]
            M length list of N length lists of datasets,
            where M is of length 2 for train+test sets, and 3 for train, val, test sets
            and N is the number of views for each.

            [[View1_train, View1_val, View1_test], [[View1_train, View1_val, View1_test]]]


        Returns
        -------
        phenonaut.data.Dataset
            _description_
        """
        datasets_train = self._phe_lists_to_df_lists(datasets_train)
        datasets_test = self._phe_lists_to_df_lists(datasets_test)

        datasets_train = self.merge_splits_to_single_views(datasets_train)

        merged_df, merged_df_feature_lists = self.multiplex_df(datasets_train, "inner")
        merged_test_df, merged_test_df_feature_lists = self.multiplex_df(
            datasets_test, "inner"
        )

        if not set(merged_df_feature_lists[0]) == set(merged_test_df_feature_lists[0]):
            raise ValueError(
                f"Training dataset(s) did not have the same features as test datasets (feature counts = {len(merged_df_feature_lists)}, and {merged_test_df_feature_lists} respectively)"
            )

        tmp_df = pd.concat([merged_df, merged_test_df])
        data_for_integration = [
            tmp_df.loc[:, merged_df_feature_lists[n]].values
            for n in range(len(merged_df_feature_lists))
        ]

        integrated_data = self.method.fit_transform(data_for_integration)
        new_features = [f"ifeat_{n+1}" for n in range(integrated_data.shape[1])]
        tmp_df.loc[:, new_features] = integrated_data
        new_ds = phenonaut.data.Dataset(
            f"{self.name}({','.join([ds.name for ds in datasets_test])})",
            tmp_df.iloc[-len(merged_test_df) :, :],
            metadata={"features": new_features},
        )
        new_ds.perturbation_column = datasets_train[0].perturbation_column
        return new_ds
