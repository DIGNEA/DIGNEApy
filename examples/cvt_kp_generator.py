#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generate_instances.py
@Time    :   2024/09/30 09:19:06
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import argparse
import itertools

import numpy as np

import digneapy as dpy
from digneapy.solvers import default_kp, map_kp, miw_kp, mpw_kp


def save_instances(filename, generated_instances, dimension):
    """Writes the generated instances into a CSV file

    Args:
        filename (str): Filename
        generated_instances (iterable): Iterable of instances
    """
    header = (
        ["target", "N"]
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
                vars = ",".join(str(x) for x in instance)
                content = solver + "," + str(dimension) + "," + vars + "\n"
                file.write(content)


def generate_instances_for_target(
    portfolio,
    dimension: int,
    generations: int = 1000,
    population_size: int = 128,
    regions: int = 1_000,
    n_samples: int = 100_000,
):
    genotype = (dimension) * 2
    cvt_centroids = np.load(f"CVTArchive_dimension_{dimension}_centroids.npy")
    cvt_samples = np.load(f"CVTArchive_dimension_{dimension}_samples.npy")
    domain = dpy.domains.KnapsackDomain(dimension=dimension)

    # Create an empty archive with the previous centroids and samples
    cvt_archive = dpy.CVTArchive(
        k=regions,
        ranges=[(1.0, 10_000), *[(1.0, 1_000) for _ in range(genotype)]],
        n_samples=n_samples,
        centroids=cvt_centroids,
        samples=cvt_samples,
    )
    map_elites = dpy.generators.MapElitesGenerator(
        domain,
        portfolio=portfolio,
        archive=cvt_archive,
        initial_pop_size=population_size,
        mutation=dpy.operators.uniform_one_mutation,
        generations=generations,
        descriptor="instance",
        repetitions=1,
    )
    archive = map_elites()
    del map_elites
    return archive


if __name__ == "__main__":
    expected_dimensions = (50, 100, 500, 1000)
    parser = argparse.ArgumentParser(
        prog="generate_instances", description="CVTMAP-Elites for KP instances"
    )
    parser.add_argument(
        "dimension",
        choices=expected_dimensions,
        type=int,
        help="dimension of the KP instances",
    )
    parser.add_argument(
        "repetition",
        choices=list(range(10)),
        type=int,
        help="Number of the run to append in the final CSV file",
    )
    args = parser.parse_args()
    n_run = args.repetition
    dimension = args.dimension

    portfolios = [
        [default_kp, map_kp, miw_kp, mpw_kp],
        [map_kp, default_kp, miw_kp, mpw_kp],
        [miw_kp, default_kp, map_kp, mpw_kp],
        [mpw_kp, default_kp, map_kp, miw_kp],
    ]

    results = {}
    for portfolio in portfolios:
        archive = generate_instances_for_target(portfolio, dimension)
        results[portfolio[0].__name__] = archive

    save_instances(
        f"kp_cvtmapelites_N_{dimension}_{n_run}.csv", results, dimension=dimension
    )
