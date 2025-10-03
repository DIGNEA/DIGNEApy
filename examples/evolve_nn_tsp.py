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

from digneapy import NS, SupportsSolve
from digneapy.archives import Archive
from digneapy.domains import TSPDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import greedy, nneighbour, two_opt
from digneapy.transformers.neural import NNEncoder
from digneapy.transformers.tuner import Tuner


class Evaluation(object):
    def __init__(
        self,
        transformer,
        domain,
        portfolios: list[list[SupportsSolve]],
    ):
        self._transformer = transformer
        self._domain = domain
        self._portfolios = portfolios

    def __call__(self, X):
        self._transformer.update_weights(X)
        results = np.zeros(4)

        for i, portfolio in enumerate(self._portfolios):
            eig = EAGenerator(
                pop_size=10,
                generations=100,
                domain=self._domain,
                portfolio=portfolio,
                novelty_approach=NS(Archive(threshold=1e-7), k=3),
                solution_set=Archive(threshold=1e-7),
                repetitions=1,
                descriptor_strategy="features",
                replacement=generational_replacement,
                transformer=self._transformer,
            )
            solutions = eig()
            results[i] = (
                -solutions.metrics["s_qd_score"]
                if solutions.metrics is not None
                else 0.0
            )

        return np.mean(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evolve NNs to encoder 11D descriptors of the TSP to 2D."
    )

    parser.add_argument(
        "-r", "--repetition", type=int, required=True, help="Repetition index."
    )
    parser.add_argument("-s", "--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()
    repetition = args.repetition
    seed = args.seed
    mp.set_start_method("spawn", force=True)

    dimension = 72
    nn = NNEncoder(
        name="NN_transformer_TSP_domain.keras",
        input_shape=[11],
        shape=(5, 2),
        activations=("relu", None),
        scale=True,
    )

    fitness = Evaluation(
        transformer=nn,
        domain=TSPDomain(dimension=50),
        portfolios=[
            [greedy, nneighbour, two_opt],
            [nneighbour, greedy, two_opt],
            [two_opt, greedy, nneighbour],
        ],
    )
    cma_es = Tuner(
        dimension=dimension, ranges=(-1.0, 1.0), generations=250, lambda_=64, seed=seed
    )
    solution = cma_es(eval_fn=fitness)
    with open(f"tsp_NN_weights_N_50_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.x))
    with open(f"tsp_fitness_NN_N_50_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.fun))

    with open(f"tsp_fitness_NN_N_50_2D_{repetition}.txt", "w") as f:
        f.write(str(solution.fun))
    # Save the model itself
    nn.update_weights(solution.x)
    nn.save(f"TSP_NN_best_transformer_N_50_to_2D_{repetition}.keras")


if __name__ == "__main__":
    main()
