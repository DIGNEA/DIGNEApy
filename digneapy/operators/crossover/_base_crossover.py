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
from typing import Optional, Protocol

import numpy as np

from digneapy.typing import IndType


# Operators
class CrossoverFn(Protocol):
    def __call__(
        self, individual: IndType, other: IndType, *args, **kwargs
    ) -> IndType: ...


class Crossover(ABC):
    def __init__(
        self,
        name: str = "CX",
        cxpb: float = 0.5,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        if cxpb < 0.0 or cxpb > 1.0:
            raise ValueError(
                f"Crossover expects cxpb in the range [0.0, 1.0]. Got {cxpb}"
            )
        self._name = name
        self._cxpb = np.float64(cxpb)
        self._seed = (
            seed
            if isinstance(seed, (int | np.random.SeedSequence))
            else np.random.SeedSequence()
        )
        self._rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        return f"{self._name}(cxpb: {self._cxpb}, seed: {self._seed.entropy})"

    def __repr__(self):
        return f"{self._name}(cxpb: {self._cxpb}, seed: {self._seed.entropy})"

    @abstractmethod
    def __call__(self, individual: IndType, other: IndType, *args, **kwargs) -> IndType:
        raise NotImplementedError("Expected to be implemented in subclasses")
