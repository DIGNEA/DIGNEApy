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

import os
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


def test_map_elites_raises_if_wrong_archive():
    with pytest.raises(TypeError) as e:
        _ = MapElites(
            KnapsackDomain(),
            portfolio=[default_kp],
            archive=tuple(),
            pop_size=100,
            mutation=BatchUMut(),
            describe_pipe=DescriptorPipeline("features"),
            repetitions=1,
        )
    assert (
        "MapElites expects an archive of class GridArchive or CVTArchive and got tuple"
        in str(e.value)
    )


@pytest.mark.parametrize("domain_cls, portfolio, feat_desc_n", DOMAIN_CONTEXT)
@pytest.mark.parametrize("dimension", ([50, 100]))
@pytest.mark.parametrize("descriptor", ("features", "performance"))
def test_map_elites_with_grid_archive(
    domain_cls, portfolio, feat_desc_n, dimension, descriptor
):
    pop_size = 32
    descriptor_pipeline = DescriptorPipeline(descriptor)
    dimension = dimension
    generations = 10
    desc_dimension = feat_desc_n if descriptor == "features" else len(portfolio)
    archive = GridArchive(
        dimensions=(10,) * desc_dimension, ranges=[(0, 1e4)] * desc_dimension
    )
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=pop_size,
        mutation=BatchUMut(),
        generations=generations,
        describe_pipe=descriptor_pipeline,
        repetitions=1,
    )
    result = map_elites()
    archive = result.instances
    assert len(map_elites.archive) == len(archive)
    assert len(archive) > 0
    assert isinstance(archive, GridArchive)
    assert all(isinstance(i, Instance) for i in archive)

    assert all(len(s.descriptor) == desc_dimension for s in archive)
    # for s in archive:
    #    assert len(s) == desc_dimension
    assert len(map_elites.log) == generations + 1

    # Is able to print the log
    log = map_elites.log
    map_elites_evolution_plot(log.logbook, "example.png")
    assert os.path.exists("example.png")
    os.remove("example.png")


test_data_cvt = [
    (
        KnapsackDomain,
        [map_kp, default_kp, miw_kp],
        [(1.0, 10_000), *[(1.0, 1_000) for _ in range(100)]],
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit],
        [(1.0, 10_000), *[(1.0, 1_000) for _ in range(50)]],
    ),
]


CVT_CONTEXT = [
    (
        KnapsackDomain,
        [default_kp, map_kp, miw_kp, mpw_kp],
    ),
    (
        BPPDomain,
        [best_fit, first_fit, worst_fit, next_fit],
    ),
    (TSPDomain, [nneighbour, greedy]),
]


def build_ranges(domain_cls, descriptor, dimension, portfolio):
    if domain_cls == KnapsackDomain:
        if descriptor == "features":
            return [(1.0, 100_000), *[(1.0, 1_000) for _ in range(7)]]
        elif descriptor == "performance":
            return [(1.0, 800_000) for _ in range(len(portfolio))]
        else:  # case instance
            return [(1.0, 100_000), *[(1.0, 1_000) for _ in range(dimension * 2)]]
    elif domain_cls == BPPDomain:
        if descriptor == "features":
            return [
                (20, 100),
                (0, 100.0),
                (20, 100),
                (20, 100),
                (20, 100),
                *[(0.0, 1.0) for _ in range(5)],
            ]
        elif descriptor == "performance":
            return [(0.0, 1.0) for _ in range(len(portfolio))]
        else:  # case instance
            return [(150, 150), *[(20, 100) for _ in range(dimension)]]
    elif domain_cls == TSPDomain:
        if descriptor == "features":
            return [(dimension * 2, dimension * 2), *[(0.0, 1_000) for _ in range(10)]]
        elif descriptor == "performance":
            return [(0.0, 1.0) for _ in range(len(portfolio))]
        else:  # case instance
            return [(0, 1_000) for _ in range(dimension * 2)]


@pytest.mark.parametrize("domain_cls, portfolio", CVT_CONTEXT)
@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_map_elites_domain_cvt(domain_cls, portfolio, descriptor):
    dimension = 100
    generations = 10
    pop_size = 32
    ranges = build_ranges(domain_cls, descriptor, dimension, portfolio)
    archive = CVTArchive(k=1000, ranges=ranges, n_samples=10_000)
    domain = domain_cls(dimension=dimension)
    assert domain.dimension == dimension

    map_elites = MapElites(
        domain,
        portfolio=portfolio,
        archive=archive,
        pop_size=pop_size,
        mutation=BatchUMut(),
        generations=generations,
        describe_pipe=DescriptorPipeline(key=descriptor),
        repetitions=1,
    )
    result = map_elites()
    archive = result.instances
    assert len(map_elites.archive) == len(archive)
    assert len(archive) != 0
    assert all(i.p >= 0 for i in archive)
    assert all(i.s == 0 for i in archive)
    assert isinstance(archive, CVTArchive)
    assert all(isinstance(i, Instance) for i in archive)
    assert len(map_elites.log) == generations + 1

    # Is able to print the log
    log = map_elites.log
    filename = Path("example.png")

    map_elites_evolution_plot(log.logbook, filename=filename)
    assert filename.exists()
    filename.unlink()
