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

from ..core import Instance, Solution
import numpy as np
import copy
from typing import Callable, Union

Crossover = Callable[
    [Union[Instance | Solution], Union[Instance | Solution]], Union[Instance | Solution]
]


def one_point_crossover(
    ind_1: Instance | Solution, ind_2: Instance | Solution
) -> Instance | Solution:
    """One point crossover

    Args:
        ind_1 Instance or Solution: First individual to apply crossover
        ind_2 Instance or Solution: Second individual to apply crossover

    Raises:
        AttributeError: When the len(ind_1) != len(ind_2)

    Returns:
        Instance or Solution: New individual
    """
    if len(ind_1) != len(ind_2):
        msg = f"Individual of different length in uniform_crossover. len(ind_1) = {len(ind_1)} != len(ind_2) = {len(ind_2)}"
        raise AttributeError(msg)

    cross_point = np.random.randint(low=0, high=len(ind_1))
    offspring = copy.deepcopy(ind_1)
    offspring[cross_point:] = ind_2[cross_point:]
    return offspring


def uniform_crossover(
    ind_1: Instance | Solution, ind_2: Instance | Solution, cxpb: float = 0.5
) -> Instance | Solution:
    """Uniform Crossover Operator for Instances and Solutions

    Args:
        ind_1 Instance or Solution: First individual to apply crossover
        ind_2 Instance or Solution: Second individual to apply crossover
        cxpb (float, optional): _description_. Defaults to 0.5.

    Raises:
        AttributeError: When the len(ind_1) != len(ind_2)

    Returns:
        Instance or Solution: New individual
    """
    if len(ind_1) != len(ind_2):
        msg = f"Individual of different length in uniform_crossover. len(ind_1) = {len(ind_1)} != len(ind_2) = {len(ind_2)}"
        raise AttributeError(msg)

    probs = np.random.rand(len(ind_1))
    offspring = copy.deepcopy(ind_1)
    chromosome = [i if pb <= cxpb else j for pb, i, j in zip(probs, ind_1, ind_2)]
    offspring[:] = chromosome
    return offspring
