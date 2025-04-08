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

__all__ = ["uniform_one_mutation", "Mutation"]

from collections.abc import Callable, Sequence
from typing import Tuple

import numpy as np

from .._core import IndType

Mutation = Callable[[IndType, Sequence[Tuple]], IndType]


def uniform_one_mutation(ind: IndType, bounds: Sequence[Tuple]) -> IndType:
    if len(ind) != len(bounds):
        msg = f"The size of individual ({len(ind)}) and bounds {len(bounds)} is different in uniform_one_mutation"
        raise ValueError(msg)

    rng = np.random.default_rng()
    mutation_point = rng.integers(low=0, high=len(ind))

    new_value = rng.uniform(
        low=bounds[mutation_point][0], high=bounds[mutation_point][1]
    )
    ind[mutation_point] = new_value
    return ind
