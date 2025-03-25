#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   crossover.py
@Time    :   2023/11/03 10:33:32
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["one_point_crossover", "uniform_crossover", "Crossover"]

from collections.abc import Callable

import numpy as np

from .._core import IndType

Crossover = Callable[
    [IndType, IndType],
    IndType,
]


def one_point_crossover(ind: IndType, other: IndType) -> IndType:
    """One point crossover

    Args:
        ind Instance or Solution: First individual to apply crossover. Returned object
        ind_2 Instance or Solution: Second individual to apply crossover

    Raises:
        ValueError: When the len(ind_1) != len(ind_2)

    Returns:
        Instance or Solution: New individual
    """
    if len(ind) != len(other):
        msg = f"Individual of different length in uniform_crossover. len(cindloned_ind) = {len(ind)} != len(ind_2) = {len(other)}"
        raise ValueError(msg)

    cross_point = np.random.randint(low=0, high=len(ind))
    ind[cross_point:] = other[cross_point:]
    return ind


def uniform_crossover(ind: IndType, other: IndType, cxpb: float = 0.5) -> IndType:
    """Uniform Crossover Operator for Instances and Solutions

    Args:
        ind Instance or Solution: First individual to apply crossover. Returned object.
        ind_2 Instance or Solution: Second individual to apply crossover
        cxpb (float, optional): _description_. Defaults to 0.5.

    Raises:
        ValueError: When the len(ind_1) != len(ind_2)

    Returns:
        Instance or Solution: New individual
    """
    if len(ind) != len(other):
        msg = f"Individual of different length in uniform_crossover. len(ind_1) = {len(ind)} != len(ind_2) = {len(other)}"
        raise ValueError(msg)

    probs = np.random.rand(len(ind))
    chromosome = [i if pb <= cxpb else j for pb, i, j in zip(probs, ind, other)]
    ind[:] = chromosome
    return ind
