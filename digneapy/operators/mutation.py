#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   mutation.py
@Time    :   2023/11/03 10:33:30
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from ..core import Instance, Solution
import numpy as np
import copy
from typing import Callable, Tuple, Union

Mutation = Callable[
    [Union[Instance | Solution], Sequence[Tuple]], Union[Instance | Solution]
]


def uniform_one_mutation(
    ind: Instance | Solution, bounds: Sequence[Tuple]
) -> Instance | Solution:
    if len(ind) != len(bounds):
        msg = "The size of individual and bounds is different in uniform_one_mutation"
        raise AttributeError(msg)
    if not all(len(b) == 2 for b in bounds):
        msg = "Error bounds in uniform_one_mutation. The bounds list must contain tuples where each tuple is (l_i, u_i)"
        raise AttributeError(msg)

    mutation_point = np.random.randint(low=0, high=len(ind))

    new_value = np.random.uniform(
        low=bounds[mutation_point][0], high=bounds[mutation_point][1]
    )
    ind[mutation_point] = new_value
    return ind
