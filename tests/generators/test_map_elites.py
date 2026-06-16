#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_map_elites.py
@Time    :   2026/04/21 14:04:19
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from pathlib import Path

import pytest

from digneapy import CVTArchive, DescriptorPipeline, GridArchive, Instance
from digneapy.domains import BPPDomain, KnapsackDomain, TSPDomain
from digneapy.generators import (
    MapElites,
)
from digneapy.operators import BatchUMut
from digneapy.solvers import (
    best_fit,
    default_kp,
    first_fit,
    greedy,
    map_kp,
    miw_kp,
    mpw_kp,
    next_fit,
    nneighbour,
    worst_fit,
)
from digneapy.visualize import map_elites_evolution_plot

DOMAIN_CONTEXT = [
    (KnapsackDomain, [default_kp, map_kp, miw_kp, mpw_kp], 8),
    (BPPDomain, [best_fit, first_fit, worst_fit, next_fit], 10),
    (TSPDomain, [nneighbour, greedy], 11),
]


@pytest.mark.parametrize("descriptor", ("features", "performance"))
def test_map_elites_generator_attrs_grid_archive(descriptor):
    number_of_items = 100
    population_size = 128
    repetitions = 1
    generations = 100
    resolution = 2
    descriptor_dim = 0
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    if descriptor == "features":
        descriptor_dim = 8
    if descriptor == "performance":
        descriptor_dim = len(portfolio)
    elif descriptor == "instance":
        descriptor_dim = (number_of_items * 2) + 1

    archive = GridArchive(
        dimensions=(resolution,) * descriptor_dim, ranges=[(0.0, 1e5)] * descriptor_dim
    )
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline(key=descriptor)
    generator = MapElites(
        domain=domain,
        portfolio=portfolio,
        pop_size=population_size,
        archive=archive,
        repetitions=repetitions,
        generations=generations,
        descriptor_pipe=descriptor_pipeline,
    )
    assert isinstance(generator, MapElites)
    assert isinstance(generator.archive, GridArchive)


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_map_elites_generator_attrs_cvt_archive(descriptor):
    number_of_items = 100
    population_size = 128
    repetitions = 1
    generations = 100
    n_centroids = 100
    descriptor_dim = 0
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    if descriptor == "features":
        descriptor_dim = 8
    if descriptor == "performance":
        descriptor_dim = len(portfolio)
    elif descriptor == "instance":
        descriptor_dim = (number_of_items * 2) + 1

    archive = CVTArchive(
        dimensions=descriptor_dim,
        centroids=n_centroids,
        ranges=[(0.0, 1e5)] * descriptor_dim,
    )
    descriptor_pipeline = DescriptorPipeline(key=descriptor)
    domain = KnapsackDomain(number_of_items=number_of_items)

    generator = MapElites(
        domain=domain,
        portfolio=portfolio,
        pop_size=population_size,
        archive=archive,
        repetitions=repetitions,
        generations=generations,
        descriptor_pipe=descriptor_pipeline,
    )
    assert isinstance(generator, MapElites)
    assert isinstance(generator.archive, CVTArchive)


def test_map_elites_generator_raises_if_wrong_archive():
    with pytest.raises(TypeError):
        _ = MapElites(
            KnapsackDomain(),
            portfolio=[default_kp],
            archive=tuple(),
            pop_size=100,
            mutation=BatchUMut(),
            descriptor_pipe=DescriptorPipeline("features"),
            repetitions=1,
        )


@pytest.mark.parametrize("descriptor", ("features", "performance"))
def test_map_elites_generator_can_generate_with_grid_archive(descriptor):
    number_of_items = 100
    population_size = 128
    repetitions = 1
    generations = 100
    resolution = 2
    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]

    descriptor_dim = 0
    if descriptor == "features":
        descriptor_dim = 8
    if descriptor == "performance":
        descriptor_dim = len(portfolio)
    elif descriptor == "instance":
        descriptor_dim = (number_of_items * 2) + 1

    archive = GridArchive(
        dimensions=(resolution,) * descriptor_dim, ranges=[(0.0, 1e5)] * descriptor_dim
    )
    domain = KnapsackDomain(number_of_items=number_of_items)
    descriptor_pipeline = DescriptorPipeline(key=descriptor)
    generator = MapElites(
        domain=domain,
        portfolio=portfolio,
        pop_size=population_size,
        archive=archive,
        repetitions=repetitions,
        generations=generations,
        descriptor_pipe=descriptor_pipeline,
    )

    result = generator()
    archive = result.instances
    assert len(generator.archive) == len(archive)
    assert len(archive) > 0
    assert isinstance(archive, GridArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert all(len(x) == (number_of_items * 2) + 1 for x in archive)
    assert all(len(s.descriptor) == descriptor_dim for s in archive)
    assert len(generator.log) == generations + 1

    # Is able to print the log
    log = generator.log
    filename = Path("example_map_elites_logbook.png")
    map_elites_evolution_plot(log, filename)
    assert filename.exists()
    filename.unlink()


def build_knapsack_archive_ranges(
    number_of_items: int, descriptor: str, n_solvers: int = 4
):
    if descriptor == "features":
        return [(1.0, 100_000), *[(1.0, 1_000) for _ in range(7)]]
    elif descriptor == "performance":
        return [(1.0, 800_000) for _ in range(n_solvers)]
    else:  # case instance
        return [(1.0, 1e5), *[(1.0, 1_000) for _ in range(number_of_items * 2)]]


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_map_elites_generator_can_generate_with_cvt_archive(descriptor):
    number_of_items = 100
    population_size = 128
    repetitions = 1
    generations = 100
    n_centroids = 100

    portfolio = [default_kp, map_kp, miw_kp, mpw_kp]
    ranges = build_knapsack_archive_ranges(
        number_of_items=number_of_items, descriptor=descriptor
    )
    descriptor_dim = 0
    if descriptor == "features":
        descriptor_dim = 8
    if descriptor == "performance":
        descriptor_dim = len(portfolio)
    elif descriptor == "instance":
        descriptor_dim = (number_of_items * 2) + 1

    archive = CVTArchive(
        dimensions=descriptor_dim,
        centroids=n_centroids,
        ranges=ranges,
    )
    descriptor_pipeline = DescriptorPipeline(key=descriptor)
    domain = KnapsackDomain(number_of_items=number_of_items)

    generator = MapElites(
        domain=domain,
        portfolio=portfolio,
        pop_size=population_size,
        archive=archive,
        repetitions=repetitions,
        generations=generations,
        descriptor_pipe=descriptor_pipeline,
    )
    result = generator()
    archive = result.instances
    assert len(generator.archive) == len(archive)
    assert len(archive) != 0
    # Novelty must be zero because MapElites doesn't use
    # this attribute of the Instance class
    assert all(i.novelty == 0 for i in archive)
    assert isinstance(archive, CVTArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(generator.log) == generations + 1

    # Is able to print the log
    log = generator.log
    filename = Path("example.png")

    map_elites_evolution_plot(log, filename=filename)
    assert filename.exists()
    filename.unlink()
