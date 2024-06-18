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

from typing import Tuple

import pytest

from digneapy.core import Domain, Instance, Problem


@pytest.fixture
def initialised_domain():
    class FixturedDomain(Domain):
        def extract_features(self, instance: Instance) -> Tuple:
            return tuple()

        def extract_features_as_dict(self, instance: Instance):
            return dict()

        def from_instance(self, instance: Instance) -> Problem:
            return None

        def generate_instance(self) -> Instance:
            return Instance()

    bounds = list((0.0, 100.0) for _ in range(100))
    return FixturedDomain(name="Fixtured_Domain", dimension=100, bounds=bounds)


def test_init_domain_attrs(initialised_domain):
    assert initialised_domain.name == "Fixtured_Domain"
    assert initialised_domain.dimension == 100
    assert initialised_domain.bounds == [(0.0, 100.0) for _ in range(100)]
    assert len(initialised_domain) == initialised_domain.dimension


def test_lower_bounds_init(initialised_domain):
    assert all(initialised_domain.lower_i(i) == 0.0 for i in range(100))
    with pytest.raises(AttributeError):
        initialised_domain.lower_i(-1)
    with pytest.raises(AttributeError):
        initialised_domain.lower_i(10000)


def test_upper_bounds_init(initialised_domain):
    assert all(initialised_domain.lower_i(i) == 0.0 for i in range(100))
    with pytest.raises(AttributeError):
        initialised_domain.upper_i(-1)
    with pytest.raises(AttributeError):
        initialised_domain.upper_i(10000)
