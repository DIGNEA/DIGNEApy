#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   opoint.py
@Time    :   2026/05/21 14:21:14
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Optional

import numpy as np

from digneapy._core._types import IndType

from .base import Crossover


class OnePointCrossover(Crossover):
    def __init__(
        self, cxpb: float = 0.5, seed: Optional[int | np.random.SeedSequence] = None
    ):
        super().__init__(cxpb, seed)

    def __call__(self, individual: IndType, other: IndType) -> IndType:
        """One point crossover

        Args:
            individual Instance or Solution: First individual to apply crossover. Returned object
            other Instance or Solution: Second individual to apply crossover
            cxpb (float64, optional): Crossover probability. Not used in this operator.
            seed (Optional[int  |  np.random.SeedSequence], optional): Seed for the random number generator. Defaults to None.

        Raises:
            ValueError: When the len(ind_1) != len(ind_2)

        Returns:
            Instance or Solution: New individual
        """
        if len(individual) != len(other):
            msg = f"Individual of different length in uniform_crossover. len(ind) = {len(individual)} != len(other) = {len(other)}"
            raise ValueError(msg)

        offspring = individual.clone()
        cross_point = self._rng.integers(low=0, high=len(individual))
        offspring[cross_point:] = other[cross_point:]
        return offspring


class OPCX(OnePointCrossover):
    pass
