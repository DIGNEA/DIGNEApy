#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_tsp_domain.py
@Time    :   2025/03/05 14:37:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import os

import numpy as np
import pytest

from digneapy import Instance
from digneapy.domains.tsp import TSP, TSPDomain


@pytest.fixture
def default_tsp():
    N = 100
    _coords = np.random.default_rng(seed=42).integers(
        low=(0),
        high=(1000),
        size=(N, 2),
        dtype=int,
    )
    return TSP(nodes=N, coords=_coords)


def test_default_tsp_instance(default_tsp):
    assert len(default_tsp) == 100
    assert default_tsp._nodes == 100
    expected_repr = "TSP<n=100>"
    assert default_tsp.__repr__() == expected_repr
    # Check is able to create a file
    default_tsp.to_file()
    assert os.path.exists("instance.tsp")
    os.remove("instance.tsp")

    with pytest.raises(Exception):
        # Raises attribute error when passing an empty list
        default_tsp.evaluate([])

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_tsp.evaluate(list(range(1000)))

    with pytest.raises(Exception):
        # Raises attribute error when passing a len(list) != len(kp)
        default_tsp.evaluate(list(range(1)))


def test_default_tsp_domain():
    dimension = 100
    domain = TSPDomain(dimension)
    assert len(domain) == dimension
    assert domain._x_range == (0, 1000)
    assert domain._y_range == (0, 1000)

    assert domain.bounds == [(0, 1000) for _ in range(dimension * 2)]

    with pytest.raises(ValueError):
        TSPDomain(dimension=-1)

    with pytest.raises(ValueError):
        TSPDomain(dimension=-1)


def test_tsp_domain_to_features():
    dimension = 100
    domain = TSPDomain(dimension)
    instance = domain.generate_instance()
    features = domain.extract_features(instance)
    assert isinstance(features, tuple)
    assert len(features) == 11
    assert all(not np.isclose(f, 0.0) for f in features)
    assert features[0] == dimension


def test_bpp_domain_to_features_dict():
    dimension = 100
    domain = TSPDomain(dimension=dimension)
    instance = domain.generate_instance()
    features = domain.extract_features_as_dict(instance)
    assert isinstance(features, dict)
    assert len(features.keys()) == 11
    assert all(not np.isclose(f, 0.0) for f in features.values())
    assert features["size"] == dimension


def test_tsp_domain_to_instance():
    dimension = 100
    variables = np.random.default_rng(seed=42).integers(
        low=1, high=1000, size=(dimension, 2)
    )
    variables = variables.flatten()
    instance = Instance(variables)

    domain = TSPDomain(dimension)
    tsp_problem = domain.from_instance(instance)
    assert len(tsp_problem) == dimension
    assert len(tsp_problem._coords) == dimension
    assert len(instance) == dimension * 2  # Flattened (x, y) coords


def test_tsp_problem(default_tsp):
    solution = default_tsp.create_solution()
    assert solution[0] == solution[-1]
    assert solution[0] == 0
    assert len(solution) == len(default_tsp) + 1
