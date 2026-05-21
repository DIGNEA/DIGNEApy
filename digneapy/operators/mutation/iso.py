#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   iso.py
@Time    :   2026/05/21 14:57:14
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from .base import Mutation


class ISOLineMutation(Mutation):
    def __init__(
        self, sigma_iso: float, sigma_line: float, seed: int | np.random.SeedSequence
    ):
        super().__init__(seed)
        try:
            self._sigma_iso = float(sigma_iso)
            self._sigma_line = float(sigma_line)
        except ValueError:
            raise ValueError("sigma_iso and sigma_line must be float.")

    def __call__(
        self,
        population: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        sigma_iso: float = 0.01,
        sigma_line: float = 0.2,
    ) -> np.ndarray:
        """Performs ISO+Line mutation from Vassiliades & Mouret 2018

        Args:
            population (np.ndarray): Batch of individuals to mutate
            lb (np.ndarray): Lower bound for each dimension
            ub (np.ndarray): Upper bound for each dimension

        Raises:
            ValueError: if dimension != bounds

        Returns:
            np.ndarray: Newly mutated individuals
        """
        dimension = len(population[0])
        if len(lb) != len(ub) or dimension != len(lb):
            msg = f"The size of individuals ({dimension}) and bounds {len(lb)} is different in iso_line_mutation"
            raise ValueError(msg)
        indices = np.arange(len(population))

        parents_a = np.asarray(
            population[self._rng.choice(indices, size=len(population))], copy=True
        )
        parents_b = np.asarray(
            population[self._rng.choice(indices, size=len(population))], copy=True
        )
        iso_noise = self._rng.normal(0, sigma_iso, size=parents_a.shape)
        line_steps = self._rng.uniform(0, sigma_line, size=(len(parents_a), 1))
        direction = parents_b - parents_a
        offspring = parents_a + iso_noise + line_steps * direction
        offspring = np.clip(offspring, lb, ub)
        return offspring


class ILMut(ISOLineMutation):
    pass
