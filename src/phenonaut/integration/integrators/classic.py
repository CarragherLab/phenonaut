import phenonaut
from .base import ViewIntegrator
from mvlearn.embed import MVMDS, SplitAE
import pandas as pd
import numpy as np
import torch

__all__ = [
    "MVMDS_ViewIntegrator",
    "SplitAutoencoder_ViewIntegrator",
    "Concatenate_ViewIntegrator",
]


class MVMDS_ViewIntegrator(ViewIntegrator):
    def __init__(self, name: str = "MVMDSViewIntegrator", **kwargs) -> None:
        super().__init__(name, False, False, True)
        self.method = MVMDS(**kwargs)

    def __call__(
        self,
        datasets: list[phenonaut.data.Dataset],
    ) -> phenonaut.data.Dataset:
        return self.fit_transform(datasets=datasets)


class SplitAutoencoder_ViewIntegrator(ViewIntegrator):
    def __init__(self, name: str = "SplitAutoEncoderViewIntegrator", **kwargs) -> None:
        super().__init__(name, True, True, True)
        self.method = SplitAE(**kwargs)

    def __call__(
        self,
        datasets_train: list[phenonaut.data.Dataset],
        datasets_test: list[phenonaut.data.Dataset],
    ) -> phenonaut.data.Dataset:
        datasets_train = self._phe_lists_to_df_lists(datasets_train)
        datasets_test = self._phe_lists_to_df_lists(datasets_test)

        datasets_train = self.merge_splits_to_single_views(datasets_train)
        print(datasets_train)
        print(datasets_test)
        ## Check! Bang!
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
        self.method.fit(data_for_integration)
        del data_for_integration
        self.method.view1_encoder_.eval()
        embedding = self.method.view1_encoder_.forward(
            torch.tensor(
                merged_df[merged_df_feature_lists[0]].values.reshape(
                    1, -1, len(merged_df_feature_lists[0])
                ),
                dtype=torch.float32,
                device="cuda:0",
            )
        )
        print(embedding.shape)
        embedding = embedding.squeeze().detach().cpu().numpy()
        print(embedding.shape)
        print(len(merged_df))
        new_features = [f"ifeat_{n+1}" for n in range(embedding.shape[1])]
        print(len(new_features))
        tmp_df.loc[:, new_features] = embedding
        new_ds = phenonaut.data.Dataset(
            f"SplitAE({','.join([ds.name for ds in datasets_test])})",
            merged_df,
            metadata={"features": new_features},
        )
        new_ds.perturbation_column = datasets_train[0].perturbation_column
        return new_ds


class Concatenate_ViewIntegrator(ViewIntegrator):
    def __init__(self, name: str = "ConcatenateVieswIntegrator", **kwargs) -> None:
        super().__init__(
            name, has_fit=False, has_transform=False, has_fit_transform=False
        )
        self.method = MVMDS(**kwargs)

    def __call__(
        self, datasets: list[phenonaut.data.Dataset]
    ) -> phenonaut.data.Dataset:
        merged_df, merged_df_feature_lists = self.multiplex_df_merge_single_samples(
            datasets
        )

        new_features = [f for fl in merged_df_feature_lists for f in fl]
        merged_df.loc[:, new_features] = merged_df
        new_ds = phenonaut.data.Dataset(
            f"MVMDS({','.join([ds.name for ds in datasets])})",
            merged_df,
            metadata={"features": new_features},
        )
        new_ds.perturbation_column = datasets[0].perturbation_column
        return new_ds
