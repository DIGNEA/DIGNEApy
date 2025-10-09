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

from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pytest

from digneapy import Domain, Instance, Problem


class FixturedDomain(Domain):
    def __init__(self):
        dimension = 100
        bounds = [(0, 1000) for _ in range(dimension)]
        super().__init__(dimension=dimension, bounds=bounds)

    def extract_features(self, instances: np.ndarray) -> np.ndarray:
        return super().extract_features(instances)

    def extract_features_as_dict(
        self, instances: np.ndarray
    ) -> List[Dict[str, np.float32]]:
        return super().extract_features_as_dict(instances)

    def generate_problems_from_instances(self, instances: np.ndarray) -> List[Problem]:
        return super().generate_problems_from_instances(instances)

    def generate_instances(self, n: int = 1) -> List[Instance]:
        return super().generate_instances(n)


@pytest.fixture
def initialised_domain():
    return FixturedDomain()


def test_fixtured_domain_attrs(initialised_domain):
    assert initialised_domain.name == "Domain"
    assert initialised_domain.dimension == 100
    assert initialised_domain.bounds == [(0.0, 1000.0) for _ in range(100)]
    assert len(initialised_domain) == initialised_domain.dimension
    assert all(initialised_domain.get_bounds_at(i) == (0.0, 1000.0) for i in range(100))


def test_fixtured_domain_raises_when_invalid_calls(initialised_domain):
    with pytest.raises(ValueError):
        initialised_domain.get_bounds_at(10000)
    with pytest.raises(ValueError):
        initialised_domain.get_bounds_at(-1)

    with pytest.raises(NotImplementedError):
        initialised_domain.generate_instances()

    with pytest.raises(NotImplementedError):
        initialised_domain.extract_features(Instance())

    with pytest.raises(NotImplementedError):
        initialised_domain.extract_features_as_dict(Instance())

    with pytest.raises(NotImplementedError):
        initialised_domain.generate_problems_from_instances([Instance()])
