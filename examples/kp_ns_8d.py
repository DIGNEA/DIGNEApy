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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import copy
import itertools

from digneapy.archives import Archive
from digneapy.domains.kp import KPDomain
from digneapy.generators import EIG
from digneapy.operators.replacement import generational
from digneapy.solvers.kp import default_kp, map_kp, miw_kp
from digneapy.transformers.autoencoders import KPAE


def save_instances(filename, generated_instances, dimension):
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
    header = (
        ["target", "N"]
        + features
        + list(
            itertools.chain.from_iterable(
                [(f"w_{i}", f"p_{i}") for i in range(dimension)]
            )
        )
    )
    with open(filename, "w") as file:
        file.write(",".join(header) + "\n")
        for solver, instances in generated_instances.items():
            for instance in instances:
                assert len(instance) == (dimension * 2) + 1
                descriptor = ",".join(str(d_i) for d_i in instance.descriptor)
                vars = ",".join(str(x) for x in instance)
                content = (
                    solver + "," + str(dimension) + "," + descriptor + "," + vars + "\n"
                )
                file.write(content)


def generate_instances_heuristics_nsf(
    dim: int,
    ta: float = 1e-6,
    tss: float = 1e-6,
):
    print(
        "=" * 40 + f" Generating KP instances of N = {dim} for Heuristics " + "=" * 40
    )
    kp_domain = KPDomain(dimension=dim, capacity_approach="percentage")
    portfolios = [
        [default_kp, map_kp, miw_kp],
        [map_kp, default_kp, miw_kp],
        [miw_kp, default_kp, map_kp],
    ]
    instances = {}
    for portfolio in portfolios:
        eig = EIG(
            pop_size=10,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            archive=Archive(threshold=3),
            s_set=Archive(threshold=3),
            k=3,
            repetitions=1,
            descriptor="features",
            replacement=generational,
        )
        _, solution_set = eig(verbose=True)
        instances[portfolio[0].__name__] = copy.deepcopy(solution_set)

    return instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="kp_ns_8d_example", description="Novelty Search for KP instances."
    )
    parser.add_argument(
        "dimension",
        type=int,
        help="Dimension of the KP instances",
    )
    parser.add_argument(
        "repetition",
        type=int,
        help="Numb1er of the run to append in the final CSV file",
    )
    args = parser.parse_args()
    n_run = args.repetition
    dimension = args.dimension
    # dimensions = [50, 250, 500, 1000]
    generated_instances = generate_instances_heuristics_nsf(dim=dimension)

    save_instances(
        f"kp_nsf_N_{dimension}_gr_heuristics_{n_run}.csv",
        generated_instances,
        dimension=dimension,
    )
