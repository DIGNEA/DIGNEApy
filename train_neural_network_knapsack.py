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

import argparse
import json
import numpy as np
import joblib
import torch
import numpy.linalg as nplinalg
import scipy.stats as spstat
from digneapy.transformers.neural import KerasNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BINS = 20
LIMITS = (-1e4, 1e4)
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

# class Evaluation(object):
#     def __init__(self, transformer):
#         self._transformer = transformer
#         self._scaler = joblib.load("scaler_for_autoencoder_N_50.pkl")
#         self._dataset = np.load("knapsack_qd_N_50_all_descriptors.npy")
#         self._dataset = np.stack(self._scaler.transform(self._dataset))
        
#     def __call__(self, X):
#         self._transformer.update_weights(X)
#         y_pred = self._transformer._model.predict(
#             self._dataset, batch_size=1024, verbose=0
#         )
#         return -compute_uniform_dist_score(y_pred[:, 0], y_pred[:, 1])


def main():
    parser = argparse.ArgumentParser(
        description="Train a Neural Network to reduce Knapsack Instances to 2d encoding."
    )

    parser.add_argument(
        "-r", "--repetition", type=int, required=True, help="Repetition index."
    )
    parser.add_argument("-s", "--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()
    repetition = args.repetition
    seed = args.seed

    nn = KerasNN(
        name="NN_transformer_knapsack_domain.keras",
        input_shape=[101],
        shape=(50, 2),
        activations=("relu", None),
        loss_fn=spread_loss,
        evaluation_metric=uniform_distribution,
    )

    scaler = joblib.load("scaler_for_autoencoder_N_50.pkl")
    dataset = np.load("knapsack_qd_N_50_all_descriptors.npy")
    dataset = np.stack(scaler.transform(dataset))

    #X_train, X_test = train_test_split(dataset, train_size=0.9, random_state=seed)
    #X_train, X_val = train_test_split(X_train, train_size=0.8, random_state=seed)
    X_train = np.load("X_train_scaled_knapsack_ae_50.npy")
    X_val = np.load("X_val_scaled_knapsack_ae_50.npy")
    y_train = X_train[:, 2:3]
    y_val = X_val[:, 2:3]
    #np.save("X_train_scaled_knapsack_ae_50.npy", X_train)
    #np.save("X_test_scaled_knapsack_ae_50.npy", X_test)
    #np.save("X_val_scaled_knapsack_ae_50.npy", X_val)
        

    history = nn._model.fit(
        X_train,
        y_train,
        batch_size=1024,
        epochs=100,
        validation_data=(X_val, y_val),
    )
    nn.save()    
    nn._model.save_weights(
        f"knapsack_NN_weights_N_50_2D_{repetition}.weights.h5", "wb"
    )
    
    with open(f"knapsack_fitness_NN_N_50_2D_{repetition}.history", "wb") as f:
        f.write(json.dumps(history.history, indent=4))



if __name__ == "__main__":
    main()
