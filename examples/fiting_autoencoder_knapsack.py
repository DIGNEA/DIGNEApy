#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   nn_transformer_gecco_23.py
@Time    :   2023/11/10 14:09:41
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import torch
import numpy as np
import numpy.linalg as nplinalg
import scipy.stats as spstat
from digneapy.transformers.neural import NNEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BINS = 20
LIMITS = (-10_000, 10_000)
GRID_X = np.linspace(LIMITS[0], LIMITS[1], BINS)
GRID_Y = np.linspace(LIMITS[0], LIMITS[1], BINS)


def JSD(P, Q):
    P = P / nplinalg.norm(P, ord=1)
    Q = Q / nplinalg.norm(Q, ord=1)
    M = (P + Q) * 0.5
    return (spstat.entropy(P, M) + spstat.entropy(Q, M)) * 0.5


def compute_uniform_dist_score(x0, x1):
    hist, _, _ = np.histogram2d(x0, x1, bins=[GRID_X, GRID_Y])
    hist1D = hist.flatten()
    uni_val = 1.0 / hist1D.size
    uni_hist = np.full(hist1D.shape, uni_val)
    return 1.0 - JSD(uni_hist, hist1D)


def uniform_distribution(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return compute_uniform_dist_score(y_pred[:, 0], y_pred[:, 1])


def repulsion_loss(y_true, y_pred, eps=1e-8):
    """
    z: tensor shape [batch_size, 2]
    Encourages points to spread out by maximizing pairwise distances.
    """
    z = y_pred
    # pairwise squared distances (avoid zero diagonal)
    dist = (
        torch.cdist(z, z, p=2) + torch.eye(len(z), device=z.device) * 1e9
    )  # large diag to ignore

    # Inverse distances to encourage points not to be too close (repel)
    inv_dist = 1.0 / (dist + eps)

    # Sum of repulsion forces (want to minimize this, so points spread)
    loss = inv_dist.sum() / (len(z) * (len(z) - 1))
    return loss


def spread_loss(y_true, y_pred):
    return -torch.var(y_pred, dim=0).mean()


def uniformity_loss(y_true, y_pred):
    return spread_loss(y_true, y_pred) + repulsion_loss(y_true, y_pred)


if __name__ == "__main__":
    nn = NNEncoder(
        name="NN_transformer_knapsack_domain_N_50.keras",
        input_shape=[101],
        shape=(50, 2),
        activations=("relu", None),
        evaluation_metric=uniform_distribution,
        loss_fn=uniformity_loss,
    )
    X = np.load("knapsack_qd_N_50_all_descriptors.npy")
    X_train, X_test = train_test_split(X, train_size=0.6, random_state=42)
    X_train, X_val = train_test_split(X_train, train_size=0.5, random_state=13)
    y_train = X_train[:, 2:3]
    y_val = X_val[:, 2:3]
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, "scaler_for_autoencoder_N_50.pkl")
    nn._model.fit(
        X_train_scaled,
        y_train,
        batch_size=64,
        epochs=100,
        validation_data=(X_val_scaled, y_val),
    )
    nn.save()
