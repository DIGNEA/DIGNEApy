#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   binary.py
@Time    :   2026/05/21 15:17:30
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from operator import attrgetter
from typing import Optional

import numpy as np

from digneapy.typing import IndType

from ._base_selection import Selection


class BinarySelection(Selection):
    def __init__(
        self, attr: str = "fitness", seed: Optional[int | np.random.SeedSequence] = None
    ):
        super().__init__(seed)
        self._attr = attr

    def __call__(
        self,
        population: Sequence[IndType] | np.ndarray,
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
            idx1, idx2 = self._rng.integers(low=0, high=len(population), size=2)
            return max(population[idx1], population[idx2], key=attrgetter(self._attr))
