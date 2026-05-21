#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generational.py
@Time    :   2026/05/21 15:24:28
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


class Generational(Replacement):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        super().__init__(seed)

    def __call__(
        self, population: Sequence[IndType], offspring: Sequence[IndType]
    ) -> Sequence[IndType]:
        """Returns the offspring population as the new current population

        Args:
            population (Sequence[IndType]): Current population in the algorithm
            offspring (Sequence[IndType]): Offspring population

        Raises:
            ValueError: Raises if the sizes of the population are different

        Returns:
            Sequence[IndType]: New population
        """
        if len(population) != len(offspring):
            msg = f"The size of the current population ({len(population)}) != size of the offspring ({len(offspring)}) in generational replacement"
            raise ValueError(msg)

        return offspring[:]
