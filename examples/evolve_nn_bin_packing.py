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
from digneapy.domains import BPPDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import best_fit, first_fit, next_fit, worst_fit
from digneapy.transformers.neural import KerasNN
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
                novelty_approach=NS(Archive(threshold=1e-3), k=3),
                solution_set=Archive(threshold=1e-3),
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
        description="Evolve NNs to encoder 11D descriptors of the BPP to 2D."
    )

    parser.add_argument(
        "-r", "--repetition", type=int, required=True, help="Repetition index."
    )
    parser.add_argument("-s", "--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()
    repetition = args.repetition
    seed = args.seed
    mp.set_start_method("spawn", force=True)
    dimension = 67
    nn = KerasNN(
        name="NN_transformer_bin_packing_domain.keras",
        input_shape=[10],
        shape=(5, 2),
        activations=("relu", None),
        scale=True,
    )

    fitness = Evaluation(
        transformer=nn,
        domain=BPPDomain(
            dimension=120,
            min_i=20,
            max_i=100,
            max_capacity=150,
            capacity_approach="fixed",
        ),
        portfolios=[
            [best_fit, first_fit, next_fit, worst_fit],
            [first_fit, best_fit, next_fit, worst_fit],
            [next_fit, best_fit, first_fit, worst_fit],
            [worst_fit, best_fit, first_fit, next_fit],
        ],
    )
    cma_es = Tuner(
        dimension=dimension, ranges=(-1, 1.0), generations=250, lambda_=64, seed=seed
    )

    solution = cma_es(eval_fn=fitness)
    with open(f"bin_packing_NN_weights_N_120_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.x))
    with open(f"bin_packing_fitness_NN_N_120_2D_{repetition}.npy", "wb") as f:
        np.save(f, np.asarray(solution.fun))

    with open(f"bin_packing_fitness_NN_N_120_2D_{repetition}.txt", "w") as f:
        f.write(str(solution.fun))

    # Save the model itself
    nn.update_weights(solution.x)
    nn.save(f"BP_NN_best_transformer_N_120_to_2D_{repetition}.keras")


if __name__ == "__main__":
    main()
