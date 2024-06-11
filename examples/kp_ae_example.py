#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kp_ae_example.py
@Time    :   2024/05/28 10:23:57
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import copy
import itertools
from collections import deque

from digneapy.archives import Archive
from digneapy.domains.knapsack import KPDomain
from digneapy.generators import EIG
from digneapy.operators.replacement import generational
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.transformers import KPAE


def save_instances(filename, generated_instances, dimension):
    """Writes the generated instances into a CSV file

    Args:
        filename (str): Filename
        generated_instances (iterable): Iterable of instances
    """
    old_features = [
        "capacity",
        "max_p",
        "max_w",
        "min_p",
        "min_w",
        "avg_eff",
        "mean",
        "std",
    ]
    header = ["target", "N"] + list(
        itertools.chain.from_iterable(
            [(f"w_{i}", f"p_{i}") for i in range(dimension)]
        )
    )
    with open(filename, "w") as file:
        file.write(",".join(header) + "\n")
        for solver, instances in generated_instances.items():
            for instance in instances:
                content = (
                    solver
                    + ","
                    + str(dimension)
                    + ","
                    + ",".join(str(x) for x in instance)
                    + "\n"
                )
                file.write(content)


def generate_instances(dim: int = 50):
    print("=" * 40 + f" Generating KP instances of N = {dim} " + "=" * 40)
    kp_domain = KPDomain(dimension=dim, capacity_approach="percentage")
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    autoencoder = KPAE(encoding="Best")

    instances = {}
    for i in range(len(portfolio)):
        portfolio.rotate(i)  # This allow us to change the target on the fly
        eig = EIG(
            pop_size=10,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            archive=Archive(threshold=1e-5),
            s_set=Archive(threshold=1e-5),
            k=3,
            repetitions=1,
            descriptor="instance",
            replacement=generational,
            transformer=autoencoder,
        )
        _, solution_set = eig()
        instances[portfolio[0].__name__] = copy.copy(solution_set)

    save_instances(
        f"kp_ns_best_autoencoder_N_{dim}_gr.csv", instances, dimension=dim
    )


if __name__ == "__main__":
    for dimension in range(50, 1050, 50):
        generate_instances(dimension)
