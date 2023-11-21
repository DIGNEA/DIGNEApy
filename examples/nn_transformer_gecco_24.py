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
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from typing import List, Dict
from collections import deque
from digneapy.transformers import HyperCMA, NN
from digneapy.generator import EIG
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement
import numpy as np
import pandas as pd


class NSEval:
    def __init__(self, features_info: Dict, resolution: int = 20):
        self.resolution = resolution
        self.features_info = features_info
        self.hypercube = [
            np.linspace(start, stop, self.resolution) for start, stop in features_info
        ]
        self.kp_domain = KPDomain(dimension=50, capacity_approach="percentage")
        self.portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])

    def __call__(self, transformer: NN):
        gen_instances = []
        eig = EIG(
            10,
            10000,
            domain=self.kp_domain,
            portfolio=self.portfolio,
            t_a=3,
            t_ss=3,
            k=3,
            repetitions=1,
            descriptor="features",
            replacement=first_improve_replacement,
            transformer=transformer,
        )
        for i in range(len(self.portfolio)):
            self.portfolio.rotate(i)
            _, solution_set = eig()
            descriptors = [list(i.features) for i in solution_set]
            gen_instances.extend(descriptors)

        # Combinar las instancias
        # Calcular el cubrimiento con respecto al espacio de referencia
        coverage = {k: set() for k in range(8)}
        for descriptor in gen_instances:
            for i, f in enumerate(descriptor):
                coverage[i].add(np.digitize(f, self.hypercube[i]))

        f = sum(len(s) for s in coverage.values())
        return f


def main():
    R = 20
    dimension = 118
    nn = NN(
        name="NN_transformer_kp_domain.keras",
        shape=(8, 4, 2),
        activations=("relu", "relu", None),
    )
    features_info = [
        (711, 30000),
        (890, 1000),
        (860, 1000.0),
        (1.0, 200),
        (1.0, 230.0),
        (0.10, 12.0),
        (400, 610),
        (240, 330),
    ]
    ns_eval = NSEval(features_info, resolution=R)
    cma_es = HyperCMA(
        dimension=dimension,
        direction="maximise",
        lambda_=32,
        generations=100,
        transformer=nn,
        eval_fn=ns_eval,
    )
    best_nn_weights, population, logbook = cma_es()
    nn.update_weights(best_nn_weights)
    nn.save()


if __name__ == "__main__":
    main()
