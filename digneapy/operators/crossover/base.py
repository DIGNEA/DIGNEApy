#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   base.py
@Time    :   2026/05/21 14:34:53
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from digneapy._core._types import IndType


class Crossover(ABC):
    def __init__(
        self, cxpb: float, seed: Optional[int | np.random.SeedSequence], *args, **kwargs
    ):
        self._cxpb = np.float64(cxpb)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def __call__(self, individual: IndType, other: IndType, *args, **kwargs) -> IndType:
        raise NotImplementedError("Expected to be implemented in subclasses")
