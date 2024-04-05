# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class _MRegressorNN(nn.Module):
    def __init__(self, network_shape):
        super().__init__()
        self.layers = nn.ModuleList()
        for nodes_i in range(len(network_shape) - 1):
            self.layers.append(
                nn.Linear(network_shape[nodes_i], network_shape[nodes_i + 1])
            )
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(network_shape[-2], network_shape[-1]))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return x


class MultiRegressorNN:
    def __init__(
        self,
        batch_size=128,
        learning_rate=1e-3,
        epochs=100,
        num_hidden_layers=1,
        hidden_layer_sizes: list[int] | None = None,
        use_optimizer: str = "ADAM",
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_hidden_layers = num_hidden_layers
        self.use_optimizer = use_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layer_sizes = hidden_layer_sizes
        if self.hidden_layer_sizes is None:
            self.network_shape = np.linspace(
                100, 100, num=self.num_hidden_layers + 2, dtype=int
            )
        else:
            self.network_shape = np.array([100] + self.hidden_layer_sizes + [100])
        self.model = _MRegressorNN(self.network_shape).to(self.device)

    def fit(self, X, y):
        if self.network_shape[0] != X.shape[1] or self.network_shape[-1] != y.shape[1]:
            del self.model
            if self.hidden_layer_sizes is None:
                self.network_shape = np.linspace(
                    X.shape[1], y.shape[1], num=self.num_hidden_layers + 2, dtype=int
                )
            else:
                self.network_shape = np.array(
                    [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
                )
            self.model = _MRegressorNN(self.network_shape).to(self.device)
        y = torch.tensor(y.astype(np.float32))
        X = torch.tensor(X.astype(np.float32))
        train_tensor = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_tensor, batch_size=self.batch_size, shuffle=True
        )

        optimizer = None
        if self.use_optimizer == "ADAM":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.use_optimizer == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(
                f"Valid options for use_optimizer argument are 'ADAM' or 'SGD', use_optimizer was {self.use_optimizer}"
            )
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                X_train, y_train = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(X_train.to(self.device))
                loss = torch.sqrt(criterion(outputs, y_train.to(self.device)))
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        test_loss = 0
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            data = X.to(self.device)
            return self.model(data).to("cpu").numpy()
