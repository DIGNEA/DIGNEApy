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

import argparse
import copy
import itertools

from digneapy import Archive, NS
from digneapy.domains import KnapsackDomain
from digneapy.generators import EAGenerator
from digneapy.operators import generational_replacement
from digneapy.solvers import default_kp, map_kp, miw_kp
from digneapy.transformers.autoencoders import KPEncoder


def save_instances(filename, generated_instances, dimension, encoding):
    """Writes the generated instances into a CSV file

    Args:
        filename (str): Filename
        generated_instances (iterable): Iterable of instances
    """
    desc_header = list(f"x_{i}" for i in range(2))
    header = (
        ["target", "N"]
        + desc_header
        + ["capacity"]
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


def generate_instances_heuristics(
    dim: int,
    encoder: str,
    ta: float = 1e-5,
    tss: float = 1e-5,
):
    print(
        "=" * 40
        + f" Generating KP instances of N = {dim} for Heuristics with encoder {encoder} "
        + "=" * 40
    )
    kp_domain = KnapsackDomain(dimension=dim, capacity_approach="percentage")
    autoencoder = KPEncoder(encoder=encoder)

    portfolios = [
        [default_kp, map_kp, miw_kp],
        [map_kp, default_kp, miw_kp],
        [miw_kp, default_kp, map_kp],
    ]
    instances = {}
    for portfolio in portfolios:
        p_names = [s.__name__ for s in portfolio]
        status = f"\rRunning portfolio: {p_names}"
        print(status, end="")
        eig = EAGenerator(
            pop_size=10,
            generations=1000,
            domain=kp_domain,
            portfolio=portfolio,
            novelty_approach=NS(Archive(threshold=ta), k=3, dist_metric="cosine"),
            solution_set=NS(Archive(threshold=tss), k=1, dist_metric='cosine'),
            repetitions=1,
            descriptor_strategy="instance",
            replacement=generational_replacement,
            transformer=autoencoder,
        )
        _, solution_set = eig(verbose=False)
        instances[portfolio[0].__name__] = copy.deepcopy(solution_set)

    # When completed clear the terminal
    blank = " " * 80
    print(f"\r{blank}\r", end="")
    return instances


if __name__ == "__main__":
    expected_dimensions = (50, 100, 500, 1000)

    parser = argparse.ArgumentParser(
        prog="variable_autoencoder_kp",
        description="Novelty Search for KP instances with variable size AE",
    )
    parser.add_argument(
        "dimension",
        choices=expected_dimensions,
        type=int,
        help="dimension of the KP instances",
    )
    parser.add_argument(
        "repetition",
        type=int,
        help="Number of the run to append in the final CSV file",
    )
    args = parser.parse_args()
    n_run = args.repetition
    dimension = args.dimension
    encoder = "variable"
    generated_instances = generate_instances_heuristics(
        dim=dimension, encoder=encoder, ta=1e-3, tss=1e-3
    )

    save_instances(
        f"kp_ns_KPEncoder_{encoder}_cosine_N_{dimension}_gr_heuristics_{n_run}.csv",
        generated_instances,
        dimension=dimension,
        encoding=encoder,
    )
