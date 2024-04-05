# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class _DAVE_model(nn.Module):
    def _topol_to_inout_tuples(
        self, input: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, int], tuple[tuple[int, ...]]]:
        """DAVE topology to view encoder sizes tuple list, embeding tuple, and decoder tuple list

        Given a view topology like [100, 77, 62, 32]
        We need to generate nn.Linear input and output size tuples for:
        view_encoder [(100, 77), (77, 62)]
        embedding (62, 32)
        view_decoder [(32, 62), (62, 77), (77, 100)]
        The above is returned as a 3 member tuple, for encoder,embedding and decoder, like:
        ([(100, 77), (77, 62)], (62, 32), [(32, 62), (62, 77), (77, 100)])

        Parameters
        ----------
        input : np.ndarray
            View encoder node sizes, like: ([100,  77,  62,  32])

        Returns
        -------
        tuple[tuple[int, ...], tuple[int, int], tuple[tuple[int, ...]]]
            3 member tuple containing:
            element 0 = List of encoder node input output tuples
            element 1 = 2 member tuple giving embedding space input and output sizes
            element 2 = List of decoder node input output tuples
        """
        encoding = [(input[0], input[1])]
        for i in range(2, len(input) - 1):
            encoding.append((encoding[-1][1], input[i]))
        embedding = (input[-2], input[-1])
        decoding = [(embedding[1], encoding[-1][1])] + list(
            (t[1], t[0]) for t in encoding[::-1]
        )
        return (encoding, embedding, decoding)

    def _assign_layers(self, topology: tuple[tuple[int, ...], tuple[int, ...]]):
        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()

        # Set up network nodes for view 1 - also do embedding here
        encoder_dims, embedding_dims, decoder_dims = self._topol_to_inout_tuples(
            topology[0]
        )
        for dims in encoder_dims:
            self.encoder1.append(nn.Linear(*dims))
            self.encoder1.append(nn.ReLU())
        self.latent1 = nn.Linear(*embedding_dims)
        self.latent2 = nn.Linear(*embedding_dims)
        for dims in decoder_dims[:-1]:
            self.decoder1.append(nn.Linear(*dims))
            self.decoder1.append(nn.ReLU())
        self.decoder1.append(nn.Linear(*decoder_dims[-1]))
        self.decoder1.append(nn.Sigmoid())

        # Set up network nodes for view 2 - embedding already initialised above
        encoder_dims, embedding_dims, decoder_dims = self._topol_to_inout_tuples(
            topology[1]
        )
        for dims in encoder_dims:
            self.encoder2.append(nn.Linear(*dims))
            self.encoder2.append(nn.ReLU())
        for dims in decoder_dims[:-1]:
            self.decoder2.append(nn.Linear(*dims))
            self.decoder2.append(nn.ReLU())
        self.decoder2.append(nn.Linear(*decoder_dims[-1]))
        self.decoder2.append(nn.Sigmoid())

    def __init__(
        self,
        view_sizes: tuple[int, int] = (100, 100),
        embedding_size: int = 32,
        n_hidden: Union[int, tuple[int, int], list[int, int]] = (3, 3),
        topology: tuple[tuple[int, ...], tuple[int, ...]] = None,
    ):
        super().__init__()
        if isinstance(n_hidden, int):
            n_hidden = (n_hidden, n_hidden)
        if topology is None:
            view_1 = np.linspace(
                view_sizes[0], embedding_size, n_hidden[0] + 2, dtype=int
            )
            view_2 = np.linspace(
                view_sizes[1], embedding_size, n_hidden[1] + 2, dtype=int
            )
            # Num inputs to latent space must be the same for each view.
            if view_1[-2] != view_2[-2]:
                if len(view_1) < 3 or len(view_2) < 3:
                    raise ValueError(
                        "Node sizes leading into embedding space were not the same, and with no hidden layers, "
                        "the sizes could not be averaged to give view 1 and view 2 pre-embedding layer the same number of inputs"
                    )
                pre_embedding_mean_layer_size = int((view_1[-2] + view_2[-2]) / 2)
                view_1[-2] = pre_embedding_mean_layer_size
                view_2[-2] = pre_embedding_mean_layer_size

            topology = [view_1, view_2]

        if topology[0][-1] != topology[1][-1]:
            raise ValueError(
                f"topology[0] and [1] did not both converge to the same number of embedding dimensions ({topology[0][-1]} and {topology[1][-1]})"
            )
        self.topology = topology
        self._assign_layers(topology)

    def encode_view1(self, x):
        for layer in self.encoder1:
            x = layer(x)
        return self.latent1(x), self.latent2(x)

    def encode_view2(self, x):
        for layer in self.encoder2:
            x = layer(x)
        return self.latent1(x), self.latent2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_2_views(self, mu1, mu2, logvar1, logvar2):
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)
        return mu1 + mu2 + eps1 * std1 + eps2 * std2

    def decode_view1(self, z):
        for layer in self.decoder1:
            z = layer(z)
        return z

    def decode_view2(self, z):
        for layer in self.decoder2:
            z = layer(z)
        return z

    def forwardA(self, x):
        z, _ = self.encode_view1(x)
        return self.decode_view1(z), self.decode_view2(z)

    def forwardB(self, x):
        z, _ = self.encode_view2(x)
        return self.decode_view1(z), self.decode_view2(z)

    def forward(self, xA, xB):
        muA, logvarA = self.encode_view1(xA)
        muB, logvarB = self.encode_view2(xB)
        z = self.reparameterize_2_views(muA, muB, logvarA, logvarB)
        return self.decode_view1(z), self.decode_view2(z), muA, muB, logvarA, logvarB


class DAVE:
    def __init__(
        self,
        batch_size=1024,
        learning_rate=1e-3,
        epochs=10,
        num_hidden_layers=1,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_hidden_layers = num_hidden_layers
        self.model = _DAVE_model()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_XA, recon_XB, XA, XB, muA, muB, logvarA, logvarB):
        bceA = F.binary_cross_entropy(recon_XA, XA, reduction="sum")
        bceB = F.binary_cross_entropy(recon_XB, XB, reduction="sum")
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kldA = -0.5 * torch.sum(1 + logvarA - muA.pow(2) - logvarA.exp())
        kldB = -0.5 * torch.sum(1 + logvarB - muB.pow(2) - logvarB.exp())
        return bceA + bceB + kldA + kldB

    def fit(self, X, y):
        if (
            self.model.topology[0][0] != X.shape[1]
            or self.model.topology[1][0] != y.shape[1]
        ):
            self.model = _DAVE_model(
                view_sizes=(X.shape[1], y.shape[1]), n_hidden=self.num_hidden_layers
            ).to(self.device)
        print(
            f"Constructed DAVE model on {self.device}, {self.batch_size=}, {self.learning_rate=}, {self.epochs=}, {self.num_hidden_layers=}"
        )
        y = torch.tensor(y.astype(np.float32))
        X = torch.tensor(X.astype(np.float32))
        train_tensor = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_tensor, batch_size=self.batch_size, shuffle=True
        )

        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                X_train, y_train = data
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                reconA, reconB, muA, muB, logvarA, logvarB = self.model(
                    X_train, y_train
                )
                loss = self.loss_function(
                    reconA, reconB, X_train, y_train, muA, muB, logvarA, logvarB
                )
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(X.astype(np.float32)).to(self.device)
            return (
                self.model.decode_view2(self.model.encode_view1(data)[0])
                .to("cpu")
                .numpy()
            )
