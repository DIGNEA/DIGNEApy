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

from digneapy.autoencoders import KPAE
from collections import deque
from digneapy.generator import EIG
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp
from digneapy.domains.knapsack import KPDomain
from digneapy.operators.replacement import first_improve_replacement
import copy


def save_instances(filename, generated_instances):
    """Writes the generated instances into a CSV file

    Args:
        filename (str): Filename
        generated_instances (iterable): Iterable of instances
    """
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
        for solver, instances in generated_instances.items():
            for inst in instances:
                content = solver + "," + ",".join(str(f) for f in inst.features) + "\n"
                file.write(content)


def main(dim: int = 500):
    kp_domain = KPDomain(dimension=dim, capacity_approach="percentage")
    portfolio = deque([default_kp, map_kp, miw_kp, mpw_kp])
    autoencoder = KPAE(encoding="Best")

    instances = {}
    for i in range(len(portfolio)):
        portfolio.rotate(i)  # This allow us to change the target on the fly
        eig = EIG(
            10,
            1000,
            domain=kp_domain,
            portfolio=portfolio,
            t_a=1e-5,
            t_ss=1e-4,
            k=3,
            repetitions=1,
            descriptor="instance",
            replacement=first_improve_replacement,
            transformer=autoencoder,
        )
        print(eig)
        _, solution_set = eig()
        print(solution_set)
        instances[portfolio[0].__name__] = copy.copy(solution_set)


if __name__ == "__main__":
    main()
