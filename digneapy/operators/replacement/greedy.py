#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   greedy.py
@Time    :   2026/05/21 15:26:26
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from digneapy._core._types import IndType

from .base import Replacement


class GreedyReplacement(Replacement):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        super().__init__(seed)

    def __call__(
        self,
        population: Sequence[IndType],
        offspring: Sequence[IndType],
    ) -> Sequence[IndType]:
        """Returns a new population produced by a greedy operator.
        Each individual in the current population is compared with its analogous in the offspring population
        and the best survives

        Args:
            population (Sequence[IndType]): Current population in the algorithm
            offspring (Sequence[IndType]): Offspring population

        Raises:
            ValueError: Raises if the sizes of the population are different

        Returns:
            Sequence[IndType]: New population
        """
        if len(population) != len(offspring):
            msg = f"The size of the current population ({len(population)}) != size of the offspring ({len(offspring)}) in first_improve_replacement"
            raise ValueError(msg)
        return [a if a > b else b for a, b in zip(population, offspring)]
