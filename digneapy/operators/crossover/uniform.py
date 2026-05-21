#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   uniform.py
@Time    :   2026/05/21 14:23:37
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


class UniformCrossover(Crossover):
    def __init__(self, cxpb: float, seed: Optional[int | np.random.SeedSequence]):
        super().__init__(cxpb, seed)

    def __call__(self, individual: IndType, other: IndType) -> IndType:
        """Uniform Crossover Operator for Instances and Solutions

        Args:
            individual (IndType): First individual to apply crossover. Returned object.
            other (IndType): Second individual to apply crossover
            cxpb (float64, optional): Crossover probability. Defaults to 0.5.
            seed (Optional[int  |  np.random.SeedSequence], optional): Seed for the random number generator. Defaults to None.

        Raises:
            ValueError: When the len(ind_1) != len(ind_2)

        Returns:
            ndarray: New individual
        """

        if len(individual) != len(other):
            msg = f"Individual of different length in uniform_crossover. len(ind) = {len(individual)} != len(other) = {len(other)}"
            raise ValueError(msg)

        probs = self._rng.random(size=len(individual))
        genotype = np.empty_like(individual)
        genotype = np.where(probs <= self._cxpb, individual, other)

        return individual.__class__(genotype)


class UCX(UniformCrossover):
    pass
