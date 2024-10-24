#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   selection.py
@Time    :   2023/11/03 10:33:26
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["binary_tournament_selection", "Selection"]

import copy
from collections.abc import Callable, Sequence

import numpy as np

from .._core import IndType

Selection = Callable[[Sequence[IndType]], IndType]


def binary_tournament_selection(
    population: Sequence[IndType],
) -> IndType:
    """Binary Tournament Selection Operator

    Args:
        population (Sequence): Population of individuals to select a parent from

    Raises:
        RuntimeError: If the population is empty

    Returns:
        Instance or Solution: New parent
    """
    if not population:
        msg = "Trying to selection individuals in an empty population."
        raise RuntimeError(msg)
    if len(population) == 1:
        return copy.deepcopy(population[0])
    else:
        idx1, idx2 = np.random.randint(low=0, high=len(population), size=2)
        ind_1 = population[idx1]
        ind_2 = population[idx2]
        parent = copy.deepcopy(ind_1) if ind_1 > ind_2 else copy.deepcopy(ind_2)
        return parent
