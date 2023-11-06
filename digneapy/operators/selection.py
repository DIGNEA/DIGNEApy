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
from ..core import Instance, Solution
import numpy as np
import copy
from typing import Callable, List, Tuple

Selection = Callable[[List[Instance] | List[Solution]], Instance | Solution]


def binary_tournament_selection(
    population: List[Instance] | List[Solution],
) -> Instance | Solution:
    """Binary Tournament Selection Operator

    Args:
        population (Iterable): Population of individuals to select a parent from

    Raises:
        RuntimeError: If the population is empty

    Returns:
        Instance or Solution: New parent
    """
    if not population:
        msg = "Trying to selection individuals in an empty population."
        raise RuntimeError(msg)

    ind_1 = population[np.random.randint(low=0, high=len(population))]
    ind_2 = population[np.random.randint(low=0, high=len(population))]
    parent = copy.deepcopy(ind_1) if ind_1 > ind_2 else copy.deepcopy(ind_2)
    return parent
