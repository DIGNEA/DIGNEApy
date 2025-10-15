#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   autoencoders.py
@Time    :   2024/05/28 10:06:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["VAE", "KPEncoder", "KPDecoder"]

from pathlib import Path
import numpy as np
import numpy.typing as npt
from digneapy.transformers._base import Transformer
from typing import Any, Tuple
from scipy.stats import lognorm
import h5py
import torch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_PATH = Path(__file__).parent / "models/"
AUTOENCODER_NAME = "variational_autoencoder_qd_instances_N_50_scripted.pt"
VAEOutput = namedtuple("VAEOutput", ["output", "codings_mean", "codings_logvar"])


def vae_loss(y_pred, y_target, kl_weight=1.0):
    output, mean, logvar = y_pred
    kl_div = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.square(), dim=-1)
    return F.mse_loss(output, y_target) + kl_weight * kl_div.mean() / 101


class VAE(nn.Module):
    def __init__(self, codings_dim: int = 2):
        super(VAE, self).__init__()
        self.codings_dim = codings_dim
        self.encoder = nn.Sequential(
            nn.Linear(101, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 2 * codings_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(codings_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 50),
            nn.ReLU(),
            nn.Linear(50, 101),
        )

    def encode(self, X):
        return self.encoder(X).chunk(2, dim=-1)  # returns (mean, logvar)

    def sample_codings(self, codings_mean, codings_logvar):
        codings_std = torch.exp(0.5 * codings_logvar)
        noise = torch.randn_like(codings_std)
        return codings_mean + noise * codings_std

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        codings_mean, codings_logvar = self.encode(X)
        codings = self.sample_codings(codings_mean, codings_logvar)
        output = self.decode(codings)
        return VAEOutput(output, codings_mean, codings_logvar)


class KPEncoder(Transformer):
    def __init__(self, name: str = "KPEncoder"):
        super().__init__(name)

        self._expected_input_dim = 101
        self._encoder = torch.jit.load(
            MODELS_PATH / AUTOENCODER_NAME, map_location=torch.device(DEVICE)
        )

    @property
    def latent_dimension(self) -> int:
        return 2

    @property
    def expected_input_dim(self) -> int:
        return self._expected_input_dim

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        """Encodes a numpy array of 50d-KP instances into 2D encodings.

        Args:
            X (npt.NDArray): A numpy array with the definitions of the KP instances. Expected to be of shape (M, 101).

        Raises:
            ValueError: If the shape of X does not match (M, 101)

        Returns:
            np.ndarray: _description_
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self._expected_input_dim:
            raise ValueError(
                f"Expected a np.ndarray with shape (M, {self._expected_input_dim}). Instead got: {X.shape}"
            )
        codings_means, codings_log_var = self._encoder.encode(
            torch.tensor(X, device=DEVICE, dtype=torch.float32)
        )
        codings = self._encoder.sample_codings(codings_means, codings_log_var)
        # Mean and Logarithm of the variance
        return codings.cpu().detach().numpy()


class KPDecoder(Transformer):
    def __init__(self, name: str = "KPDecoder", scale_method: str = "learnt"):
        super().__init__(name)
        if scale_method not in ("learnt", "sample"):
            raise ValueError(
                "KPDecoder expects the scale method to be either learnt or sample"
            )
        self._scale_method = scale_method
        self.__scales_fname = "scales_knapsack_N_50.h5"
        self._expected_latent_dim = 2
        self._decoder = torch.jit.load(
            MODELS_PATH / AUTOENCODER_NAME, map_location=torch.device(DEVICE)
        )
        with h5py.File(MODELS_PATH / self.__scales_fname, "r") as file:
            self._max_weights = file["scales"]["max_weights"][:].astype(np.int32)
            self._max_profits = file["scales"]["max_profits"][:].astype(np.int32)
            self._sum_of_weights = file["scales"]["sum_of_weights"][:].astype(np.int32)
        if self._scale_method == "sample":
            self._weights_fitted_dist = lognorm.fit(self._max_weights, floc=0)
            self._profits_fitted_dist = lognorm.fit(self._max_profits, floc=0)
            self._capacity_fitted_dist = lognorm.fit(self._sum_of_weights, floc=0)

    @property
    def output_dimension(self) -> int:
        return 101

    def __sample_scaling_factors(self, size: int) -> Tuple[Any, Any, Any]:
        return (
            lognorm.rvs(
                self._weights_fitted_dist[0],
                loc=self._weights_fitted_dist[1],
                scale=self._weights_fitted_dist[2],
                size=size,
            )[:, None],
            lognorm.rvs(
                self._profits_fitted_dist[0],
                loc=self._profits_fitted_dist[1],
                scale=self._profits_fitted_dist[2],
                size=size,
            )[:, None],
            lognorm.rvs(
                self._capacity_fitted_dist[0],
                loc=self._capacity_fitted_dist[1],
                scale=self._capacity_fitted_dist[2],
                size=size,
            )[:, None],
        )

    def __scaling_from_training(
        self, size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indexes = np.random.randint(low=0, high=len(self._max_weights), size=size)
        return (
            self._max_weights[indexes],
            self._max_profits[indexes],
            self._sum_of_weights[indexes],
        )

    def __denormalise_instances(self, decode_X: np.ndarray) -> np.ndarray:
        n_instances = decode_X.shape[0]
        if self._scale_method == "sample":
            max_w, max_p, scale_Q = self.__sample_scaling_factors(size=n_instances)
        else:
            max_w, max_p, scale_Q = self.__scaling_from_training(size=n_instances)

        rescaled_instances = np.zeros_like(decode_X, dtype=np.int32)
        rescaled_instances[:, 0] = decode_X[:, 0] * scale_Q[:, 0]  # * 1_000_000
        rescaled_instances[:, 1::2] = decode_X[:, 1::2] * max_w  # * 100_000
        rescaled_instances[:, 2::2] = decode_X[:, 2::2] * max_p  # * 100_000
        return rescaled_instances

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        """Decodes an np.ndarray of shape (M, 2) into KP instances of N = 50.
        It does not return Knapsack objects but a np.ndarray of shape (M, 101)
        where 101 corresponds to the capacity (Q) and 50 pairs of weights and profits(w_i, p_i)

        Args:
            X (npt.NDArray): an np.ndarray of shape (M, 2)

        Raises:
            ValueError: If X has a difference shape than (M, 2)

        Returns:
            np.ndarray: numpy array with |M| KP definitions
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self._expected_latent_dim:
            raise ValueError(
                f"Expected a np.ndarray with shape (M, {self._expected_latent_dim}). Instead got: {X.shape}"
            )
        y = (
            self._decoder.decode(torch.tensor(X, device=DEVICE, dtype=torch.float32))
            .cpu()
            .detach()
            .numpy()
        )
        return self.__denormalise_instances(y)
