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
import multiprocessing as mp

import numpy as np
import joblib
import numpy.linalg as nplinalg
import scipy.stats as spstat
from digneapy.transformers.neural import NNEncoder
from digneapy.transformers.tuner import Tuner

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


class Evaluation(object):
    def __init__(self, transformer):
        self._transformer = transformer
        self._scaler = joblib.load("scaler_for_autoencoder_N_50.pkl")
        self._dataset = np.load("knapsack_qd_N_50_all_descriptors.npy")
        self._dataset = np.stack(self._scaler.transform(self._dataset))
        print(self._dataset)
        input()

    def __call__(self, X):
        self._transformer.update_weights(X)
        y_pred = self._transformer._model.predict(
            self._dataset, batch_size=1024, verbose=2
        )
        return -compute_uniform_dist_score(y_pred[:, 0], y_pred[:, 1])


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
    mp.set_start_method("spawn", force=True)

    dimension = 5202
    nn = NNEncoder(
        name="NN_transformer_knapsack_domain.keras",
        input_shape=[101],
        shape=(50, 2),
        activations=("relu", None),
    )

    fitness = Evaluation(transformer=nn)
    cma_es = Tuner(
        dimension=dimension,
        ranges=(-1.0, 1.0),
        generations=100,
        lambda_=32,
        seed=seed,
        workers=4,
    )

    solution = cma_es(eval_fn=fitness)
    with open(f"knapsack_NN_weights_N_50_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.x))
    with open(f"knapsack_fitness_NN_N_50_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.fun))

    with open(f"knapsack_fitness_NN_N_50_2D_{repetition}.txt", "w") as f:
        f.write(str(solution.fun))

    # Save the model itself
    nn.update_weights(solution.x)
    nn.save(f"KP_NN_best_transformer_N_50_to_2D_{repetition}.keras")


if __name__ == "__main__":
    main()
