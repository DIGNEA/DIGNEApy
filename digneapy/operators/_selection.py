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

from collections.abc import Callable, Sequence
from operator import attrgetter

import numpy as np

from .._core import IndType

Selection = Callable[[Sequence[IndType]], IndType]


def binary_tournament_selection(population: Sequence[IndType]) -> IndType:
    """Binary Tournament Selection Operator

    Args:
        population (Sequence): Population of individuals to select a parent from

    Raises:
        RuntimeError: If the population is empty

    Returns:
        Instance or Solution: New parent
    """
    if population is None or len(population) == 0:
        msg = "Trying to selection individuals in an empty population."
        raise ValueError(msg)
    if len(population) == 1:
        return population[0]
    else:
        idx1, idx2 = np.random.default_rng().integers(
            low=0, high=len(population), size=2
        )
        return max(population[idx1], population[idx2], key=attrgetter("fitness"))
