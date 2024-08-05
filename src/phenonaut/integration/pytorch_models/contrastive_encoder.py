# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import datetime
from tqdm import tqdm
import pandas as pd
import itertools
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import phenonaut
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from scipy.spatial.distance import cdist
import torch


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MultiLayerProjectionHead(nn.Module):
    def __init__(
        self, network_shape: list[int], dropout: float | None, layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.network = torch.nn.ModuleList()

        for nodes_i in range(1, len(network_shape)):
            self.network.append(
                torch.nn.Linear(network_shape[nodes_i - 1], network_shape[nodes_i])
            )
            self.network.append(torch.nn.GELU())
            is_last_layer = nodes_i == (len(network_shape) - 1)
            if is_last_layer:
                if dropout is not None:
                    self.network.append(torch.nn.Dropout(p=dropout))
                if layer_norm:
                    self.network.append(nn.LayerNorm(network_shape[-1]))
        print("Made a network that looks like:", self)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class ContrastiveEncoderDataset(torch.utils.data.Dataset):
    def ids_to_lookup_table(self, pert_ids_lists):
        allowed_ids = set.intersection(*[set(pid) for pid in pert_ids_lists])
        missing_ids = set.union(*[set(pid) for pid in pert_ids_lists]) - allowed_ids
        if len(missing_ids) > 0:
            print(
                f"Warning, the following perturbations could not be found in all datasets: {missing_ids}"
            )
        dfs = [
            pd.DataFrame([pid for pid in pidlist if pid in allowed_ids], columns=["id"])
            for pidlist in pert_ids_lists
        ]
        for i in range(len(dfs)):
            dfs[i].index.name = f"index_{i}"
            dfs[i] = dfs[i].reset_index()
        df = dfs[0][["id", "index_0"]]
        for i in range(1, len(dfs)):
            df = df.merge(dfs[i], left_on="id", right_on="id", how="outer")
        return df

    def downsample_df(
        self,
        df: pd.DataFrame,
        groupby: list[str] | str,
        features: list[str],
        n_max: int | None,
    ):
        if n_max is None:
            return df
        clean_df = pd.DataFrame()

        for name, g_df in df.groupby(groupby):
            if len(g_df) > n_max:
                distances = cdist(
                    g_df[features].median().values.reshape(1, -1), g_df[features].values
                )[0]
                indicies = np.argsort(distances)[:n_max]
                clean_df = pd.concat([clean_df, g_df.iloc[indicies]])
            else:
                clean_df = pd.concat([clean_df, g_df])
        return clean_df

    def __init__(
        self,
        datasets: list[phenonaut.data.Dataset],
        dataset_to_view_map: dict | None = None,
        n_max_same_perturbations: int | None = 4,
    ):
        if len(set([ds.perturbation_column for ds in datasets])) > 1:
            raise ValueError(
                "Datasets had different perturbation columns:",
                [ds.perturbation_column for ds in datasets],
            )
        cleaned_dfs = [
            self.downsample_df(
                ds.df, ds.perturbation_column, ds.features, n_max_same_perturbations
            )
            for ds in datasets
        ]
        self.dfs = [
            cleaned_dfs[i][datasets[i].features].values for i in range(len(datasets))
        ]
        if dataset_to_view_map is None:
            dataset_to_view_map = {d: 0 for d in range(len(datasets))}
        self.dataset_to_view_map = dataset_to_view_map
        # Create a lookup table which points to the iloc indexes of data within dataframes
        self.lookup = self.ids_to_lookup_table(
            [
                cleaned_dfs[i][datasets[i].perturbation_column]
                for i in range(len(datasets))
            ]
        )

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        return [
            torch.tensor(
                self.dfs[df_i][self.lookup.iloc[idx, df_i + 1]], dtype=torch.float32
            )
            for df_i in range(len(self.dfs))
        ]


class _ContrastiveEncoder_model(nn.Module):
    def __init__(
        self,
        network_shapes: list[list[int]],
        view_to_network_map: dict[int, int],
        model_learning_rates: list[float] | float = 1e-4,
        dropout: float = 0.0,
        layer_norm: bool = True,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(model_learning_rates, list):
            model_learning_rates = [model_learning_rates] * len(network_shapes)
        self.networks = torch.nn.ModuleList(
            [
                MultiLayerProjectionHead(
                    network_shape=ns, dropout=dropout, layer_norm=layer_norm
                )
                for ns in network_shapes
            ]
        )
        self.view_to_network_map = view_to_network_map
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.model_learning_rates = model_learning_rates

    def _multiple_embeddings_to_losses(
        self, embeddings: list[torch.Tensor]
    ) -> torch.tensor:
        loss = torch.zeros(
            embeddings[0].shape[0],
            dtype=torch.float32,
            device=embeddings[0].device,
        )
        for e1, e2 in itertools.combinations(embeddings, 2):
            logits = (e2 @ e1.T) / self.temperature
            v1_similarity = e1 @ e1.T
            v2_similarity = e2 @ e2.T
            targets = F.softmax(
                (v1_similarity + v2_similarity) / 2 * self.temperature, dim=-1
            )
            v1_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
            v2_loss = (-targets * self.log_softmax(logits)).sum(1)
            loss += (v1_loss + v2_loss) / 2.0
        return loss

    def forward(self, inputs):
        return [
            self.networks[self.view_to_network_map[i]](inputs[i])
            for i in range(len(inputs))
        ]

    def get_embeddings(self, x1: torch.Tensor, network_num: int):
        self.eval()
        with torch.no_grad():
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            # x1 = x1.to(self.device)
            x1 = self.networks[network_num].forward(x1)
            # for layer in self.networks[network_num]:
            #     x1 = layer(x1)
            return x1.cpu().numpy()


class ContrastiveEncoder:
    def __init__(
        self,
        datasets_train: list[phenonaut.data.Dataset],
        datasets_val: list[phenonaut.data.Dataset],
        datasets_test: list[phenonaut.data.Dataset],
        network_shapes: list[list[int]],
        view_to_network_map: dict[int, int],
        batch_size: int,
        temperature: float = 1.0,
        dropout: float | list[int] = 0.0,
        layer_norm: bool = True,
        learning_rates: float | list[float] = 1e-4,
        lr_scheduler_patience: float = 2.0,
        lr_scheduler_factor: float = 0.5,
        weight_decay=1e-3,
        repeats_as_augmentations: bool | int = False,
        num_dataloader_workers: int = -1,
        max_epochs: int = 200,
        earlystop_patience: int = 10,
        training_output_dir: str | Path | None = 'runs',
        checkpoint_file: str | None = None,
        seed: int | None = 42,
        n_max_same_perturbations: int | None = 4,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(repeats_as_augmentations, bool):
            if repeats_as_augmentations:
                repeats_as_augmentations = 2

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if training_output_dir is not None:
            writer = SummaryWriter(f"{training_output_dir}/clip_moa_{timestamp}")
        else:
            writer = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model = _ContrastiveEncoder_model(
            network_shapes,
            view_to_network_map,
            dropout=dropout,
            layer_norm=layer_norm,
            temperature=temperature,
        )
        self.model.to(self.device)

        if checkpoint_file is None:
            self.train_loader = DataLoader(
                ContrastiveEncoderDataset(
                    datasets_train,
                    view_to_network_map,
                    n_max_same_perturbations=n_max_same_perturbations,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_dataloader_workers,
            )
            self.val_loader = DataLoader(
                ContrastiveEncoderDataset(
                    datasets_val,
                    view_to_network_map,
                    n_max_same_perturbations=n_max_same_perturbations,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_dataloader_workers,
            )
            self.test_loader = DataLoader(
                ContrastiveEncoderDataset(
                    datasets_test,
                    view_to_network_map,
                    n_max_same_perturbations=n_max_same_perturbations,
                ),
                batch_size=batch_size,
                num_workers=num_dataloader_workers,
            )

            if isinstance(learning_rates, float):
                learning_rates = [learning_rates]
                if len(network_shapes) > 1:
                    learning_rates = learning_rates * len(network_shapes)

            early_stopper = EarlyStopper(patience=earlystop_patience)

            print("len networks = ", len(self.model.networks))
            parameters = [
                {
                    "params": self.model.networks[i].parameters(),
                    "lr": learning_rates[i],
                    "weight_decay": weight_decay,
                }
                for i in range(len(self.model.networks))
            ]
            optimiser = torch.optim.AdamW(parameters)

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="min",
                patience=lr_scheduler_patience,
                factor=lr_scheduler_factor,
            )

            best_vloss = 1_000_000

            self.model.train(True)
            for epoch_i in (pbar := tqdm(range(max_epochs), desc="Training epochs")):
                # Make sure gradient tracking is on, and do a pass over the data
                tloss = self.train_one_epoch(
                    epoch_i, writer, self.train_loader, optimiser, lr_scheduler, 100
                )

                vloss = self.val_one_epoch()
                lr_scheduler.step(vloss)
                pbar.set_postfix_str(f"loss = {tloss}, val= {vloss}")

                # Log the running loss averaged per batch
                # for both training and validation
                if writer:
                    writer.add_scalars(
                        "Training loss epoch",
                        {"Training": tloss, "Validation": vloss},
                        epoch_i + 1,
                    )
                    writer.flush()

                # Track best performance, and save the model's state
                if training_output_dir:
                    if vloss < best_vloss:
                        best_vloss = vloss
                        model_path = f"{training_output_dir}/checkpoints/model_{timestamp}_{epoch_i}"
                        torch.save(self.model.state_dict(), model_path)

                self.vloss = vloss
                self.num_epochs_completed = epoch_i + 1
                if early_stopper.early_stop(vloss):
                    break
        else:
            self.model.load_state_dict(torch.load(checkpoint_file))

    def train_one_epoch(
        self,
        epoch_index,
        tb_writer,
        training_loader,
        optimizer,
        lr_scheduler,
        write_every_n_steps,
    ):

        losses = []

        self.model.train(True)
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Zero your gradients for every batch!
            data = [d.to(self.device) for d in data]

            # Make predictions for this batch
            outputs = self.model(data)

            # Compute the loss and its gradients
            loss = self.model._multiple_embeddings_to_losses(outputs).mean()
            optimizer.zero_grad()
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            losses.append(loss.item())
            if tb_writer and write_every_n_steps is not None:
                if i % write_every_n_steps == write_every_n_steps - 1:
                    tb_x = epoch_index * len(training_loader) + i + 1
                    tb_writer.add_scalar("Loss/train", np.mean(losses), tb_x)

        return np.mean(losses)

    def val_one_epoch(self):
        running_vloss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(self.val_loader):
                vdata = [d.to(self.device) for d in vdata]

                voutputs = self.model(vdata)
                vloss = self.model._multiple_embeddings_to_losses(voutputs).mean()
                running_vloss += vloss.item()
            return running_vloss / len(self.val_loader)

    def get_embedded_ds(
        self,
        ds: phenonaut.Phenonaut | phenonaut.data.Dataset | np.ndarray,
        network_number: int = 0,
    ) -> phenonaut.data.dataset:
        if isinstance(ds, phenonaut.Phenonaut):
            ds = ds.ds
        embedded_ds = ds.copy()
        embeddings = self.model.get_embeddings(
            torch.tensor(ds.data.values, dtype=torch.float32, device=self.device),
            network_num=network_number,
        )
        clip_feature_names = [f"feat_{n+1}" for n in range(embeddings.shape[1])]
        embedded_ds.df.loc[:, clip_feature_names] = embeddings
        embedded_ds.features = clip_feature_names, "Embedded using CLIP"
        return embedded_ds
