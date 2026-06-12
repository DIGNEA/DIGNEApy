#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   typing.py
@Time    :   2026/06/01 15:08:55
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Callable
from enum import IntEnum
from typing import TypeVar

import numpy as np

from ._core._instance import Instance
from ._core._solution import Solution

Ind = int | np.integer
Float = float | np.floating

"""
Individual Type in Digneapy to represent Solution and Instances in methods that can be used with both
"""
IndType = TypeVar("IndType", Instance, Solution)

"""
Performance Function type. From any sequence it calculates the performance score.
Returns:
    float: Performance score
"""
PerformanceFn = Callable[[np.ndarray], np.ndarray]


class Direction(IntEnum):
    """Direction of the optimisation for Deap-based solvers."""

    MINIMISE = -1
    MAXIMISE = 1

    @classmethod
    def _create_individual(cls, direction):
        from deap import base, creator

        match direction:
            case 1:
                creator.create("FitnessMin", base.Fitness, weights=(-1,))
                creator.create("IndMin", list, fitness=creator.FitnessMin)

            case -1:
                creator.create("FitnessMax", base.Fitness, weights=(1,))
                creator.create("IndMax", list, fitness=creator.FitnessMax)

    def __new__(cls, value):
        cls._create_individual(value)
        return int.__new__(cls, value)

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))
