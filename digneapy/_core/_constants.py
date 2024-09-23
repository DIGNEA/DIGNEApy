#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _constants.py
@Time    :   2024/06/07 11:20:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["Direction", "IndType"]

from enum import IntEnum
from typing import TypeVar

from digneapy._core._instance import Instance
from digneapy._core._solution import Solution

# from deap import base, creator

# """
#     Definition of the constant Fitness and Individual types
#     for the DEAP algorithms used in DIGNEApy.
#     These values are used in the evolutionary.py and tuner.py modules
# """
# creator.create("FitnessMax", base.Fitness, weights=(1,))
# creator.create("FitnessMin", base.Fitness, weights=(-1,))

# creator.create("IndMax", list, fitness=creator.FitnessMax)
# creator.create("IndMin", list, fitness=creator.FitnessMin)


def create_individual(direction):
    from deap import base, creator

    match direction:
        case 1:
            creator.create("FitnessMin", base.Fitness, weights=(-1,))
            creator.create("IndMin", list, fitness=creator.FitnessMin)

        case -1:
            creator.create("FitnessMax", base.Fitness, weights=(1,))
            creator.create("IndMax", list, fitness=creator.FitnessMax)


class Direction(IntEnum):
    MINIMISE = -1
    MAXIMISE = 1

    def __new__(cls, value):
        create_individual(value)
        return int.__new__(cls, value)

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


IndType = TypeVar("IndType", Instance, Solution)
