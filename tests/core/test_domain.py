#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_domain.py
@Time    :   2024/06/18 11:36:09
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Dict

import numpy as np
import pytest

from digneapy.core import Domain, Instance, Problem


class FixturedDomain(Domain):
    def __init__(self, dimension: int, low: int = 0, high: int = 1000):
        bounds = [(low, high) for _ in range(dimension)]
        super().__init__(dimension=dimension, bounds=bounds)

    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        return super().extract_features(instances)

    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> Sequence[Dict]:
        return super().extract_features_as_dict(instances)

    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> Sequence[Problem]:
        return super().generate_problems_from_instances(instances)

    def generate_instances(self, n=1) -> Sequence[Instance]:
        return super().generate_instances(n)


def test_fixtured_domain_attrs():
    dimension = 10
    domain = FixturedDomain(dimension)
    assert domain.__name__ == "Domain"
    assert domain.dimension == dimension
    assert domain.bounds == [(0.0, 1000.0) for _ in range(dimension)]
    assert len(domain) == domain.dimension
    assert all(domain.get_bounds_at(i) == (0.0, 1000.0) for i in range(dimension))

    lbs = domain.lbs
    ubs = domain.ubs
    assert len(lbs) == len(ubs)
    assert all(lbi == 0.0 for lbi in lbs)
    assert all(ubi == 1000.0 for ubi in ubs)


def test_fixtured_domain_negative_dim():
    dimension = -10
    with pytest.raises(ValueError):
        _ = FixturedDomain(dimension)


def test_domain_raises_if_out_range():
    dimension = 10
    domain = FixturedDomain(dimension=dimension)
    with pytest.raises(ValueError):
        domain.get_bounds_at(10000)

    with pytest.raises(ValueError):
        domain.get_bounds_at(-100)


def test_domain_raises_not_impl_abstract():
    dimension = 10
    domain = FixturedDomain(dimension=dimension)
    default_variables = tuple(range(10))

    with pytest.raises(NotImplementedError):
        domain.generate_instances()

    with pytest.raises(NotImplementedError):
        domain.extract_features([Instance(default_variables)])

    with pytest.raises(NotImplementedError):
        domain.extract_features_as_dict([Instance(default_variables)])

    with pytest.raises(NotImplementedError):
        domain.generate_problems_from_instances([Instance(default_variables)])
