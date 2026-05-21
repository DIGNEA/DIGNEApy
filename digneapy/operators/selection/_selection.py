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

from collections.abc import Sequence
from operator import attrgetter
from typing import Optional, Protocol

import numpy as np

from .._core import IndType


class Selection(Protocol):
    """Protocol that defines the Selection mechanism"""

    def __call__(
        self,
        population: Sequence[IndType] | np.ndarray,
        seed: Optional[int | np.random.SeedSequence] = None,
        *arg,
        **kwargs,
    ) -> IndType: ...


def binary_tournament_selection(
    population: Sequence[IndType],
    seed: Optional[int | np.random.SeedSequence] = None,
    *args,
    **kwargs,
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
        raise ValueError(msg)
    elif len(population) == 1:
        return population[0]
    else:
        idx1, idx2 = np.random.default_rng(seed).integers(
            low=0, high=len(population), size=2
        )
        return max(population[idx1], population[idx2], key=attrgetter("fitness"))
