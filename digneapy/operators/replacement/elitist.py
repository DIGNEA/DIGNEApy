#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   elitist.py
@Time    :   2026/05/21 15:27:28
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import itertools
from collections.abc import Sequence
from operator import attrgetter
from typing import Optional

import numpy as np

from digneapy.typing import IndType

from ._base_replacement import Replacement


class Elitist(Replacement):
    def __init__(
        self,
        hall_of_fame: int = 1,
        attr: str = "fitness",
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        super().__init__(seed)
        self._hof = hall_of_fame
        self._attr = attr

    def __call__(
        self,
        population: Sequence[IndType],
        offspring: Sequence[IndType],
    ) -> Sequence[IndType]:
        """Returns a new population constructed using the Elitist approach.
        HoF number of individuals from the current + offspring populations are
        kept in the new population. The remaining individuals are selected from
        the offspring population.

        Args:
            population Sequence[IndType],: Current population in the algorithm
            offspring  Sequence[IndType],: Offspring population
            hof (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: Raises if the sizes of the population are different

        Returns:
            list[IndType]:
        """
        if len(population) != len(offspring):
            msg = f"The size of the current population ({len(population)}) != size of the offspring ({len(offspring)}) in elitist_replacement"
            raise ValueError(msg)

        combined_population = sorted(
            itertools.chain(population, offspring),
            key=attrgetter(self._attr),
            reverse=True,
        )
        top = combined_population[: self._hof]
        return list(top + offspring[1:])
