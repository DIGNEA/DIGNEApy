#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_sorting.py
@Time    :   2025/10/20 15:00:31
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import pytest
from digneapy.domains import KnapsackDomain
from digneapy.utils import sort_knapsack_instances
import numpy as np


@pytest.mark.parametrize("n_instances", [10, 50, 100])
@pytest.mark.parametrize("dimension", [50, 100, 1000])
def test_sorting_knapsack_instances(n_instances, dimension):
    domain = KnapsackDomain(dimension=dimension)
    instances = domain.generate_instances(n=n_instances)
    assert len(instances) == n_instances
    instances_array = np.asarray(instances, copy=True)
    capacities = instances_array[:, 0]
    assert instances_array.shape == (n_instances, (dimension * 2) + 1)
    sorted_instances = sort_knapsack_instances(instances_array)
    assert sorted_instances is instances_array
    assert sorted_instances.shape == instances_array.shape
    np.testing.assert_equal(sorted_instances[:, 0], capacities)
    for instance in sorted_instances:
        for i_pair in range(1, dimension * 2, 4):
            left = tuple(instance[i_pair : i_pair + 2])
            right = tuple(instance[i_pair + 2 : i_pair + 4])
            assert left <= right
