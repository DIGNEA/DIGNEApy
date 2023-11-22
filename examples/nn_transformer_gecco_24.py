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
from digneapy.operators.replacement import generational, first_improve_replacement
import numpy as np
import pandas as pd


def save_instances(filename, generated_instances):
    features = [
        "target",
        "capacity",
        "max_p",
        "max_w",
        "min_p",
        "min_w",
        "avg_eff",
        "mean",
        "std",
    ]
    with open(filename, "w") as file:
        file.write(",".join(features) + "\n")
        for solver, descriptors in generated_instances.items():
            for desc in descriptors:
                content = solver + "," + ",".join(str(f) for f in desc) + "\n"
                file.write(content)


class NSEval:
    def __init__(self, features_info: Dict, resolution: int = 20):
        self.resolution = resolution
        self.features_info = features_info
        self.hypercube = [
            np.linspace(start, stop, self.resolution) for start, stop in features_info
        ]
        self.kp_domain = KPDomain(dimension=50, capacity_approach="percentage")
        self.portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])

    def __call__(self, transformer: NN, filename: str = None):
        gen_instances = {s.__name__: [] for s in self.portfolio}

        for i in range(len(self.portfolio)):
            self.portfolio.rotate(i)
            eig = EIG(
                10,
                1000,
                domain=self.kp_domain,
                portfolio=self.portfolio,
                t_a=0.5,
                t_ss=0.05,
                k=3,
                repetitions=1,
                descriptor="features",
                replacement=first_improve_replacement,
                transformer=transformer,
            )
            archive, solution_set = eig()
            descriptors = [list(i.features) for i in solution_set]
            gen_instances[self.portfolio[0].__name__].extend(descriptors)
        if any(len(l) != 0 for l in gen_instances.values()):
            save_instances(filename, gen_instances)
        # Combinar las instancias
        # Calcular el cubrimiento con respecto al espacio de referencia
        coverage = {k: set() for k in range(8)}
        for solver, descriptors in gen_instances.items():  # For each set of instances
            for desc in descriptors:  # For each descriptor in the set
                for i, f in enumerate(desc):  # Location of the ith feature
                    digits = np.digitize(f, self.hypercube[i])
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
        lambda_=10,
        generations=50,
        transformer=nn,
        eval_fn=ns_eval,
    )
    best_nn_weights, population, logbook = cma_es()
    nn.update_weights(best_nn_weights)
    nn.save("NN_best_transformer_found_kp_domain.keras")
    for i, ind in enumerate(population):
        nn.update_weights(ind)
        nn.save(f"NN_final_population_{i}_transformer_kp_domain.keras")

    # Saving logbook to CSV
    df = pd.DataFrame(logbook)
    df.to_csv("CMAES_logbook_for_NN_transformers.csv", index=False)


if __name__ == "__main__":
    main()
