#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   base.py
@Time    :   2026/05/21 14:47:58
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from digneapy.typing import IndType


class Mutation(ABC):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        self._seed = (
            seed
            if isinstance(seed, (int | np.random.SeedSequence))
            else np.random.SeedSequence()
        )
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def __call__(
        self,
        population: IndType | np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Expected to be implemented in subclasses")
