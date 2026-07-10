#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   base.py
@Time    :   2026/05/21 15:23:37
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC
from collections.abc import Sequence
from typing import Optional, Protocol

import numpy as np

from digneapy.typing import IndType


class ReplacementFn(Protocol):
    def __call__(
        self,
        population: Sequence[IndType],
        offspring: Sequence[IndType],
        *arg,
        **kwargs,
    ) -> Sequence[IndType]: ...


class Replacement(ABC):
    def __init__(
        self,
        name: str = "Replacement",
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        self._seed = (
            seed
            if isinstance(seed, (int | np.random.SeedSequence))
            else np.random.SeedSequence()
        )
        self._rng = np.random.default_rng(seed)
        self._name = name

    def __str__(self) -> str:
        return f"{self._name}(seed: {self._seed})"

    def __repr__(self):
        return f"{self._name}(seed: {self._seed})"

    def __call__(
        self,
        population: Sequence[IndType],
        offspring: Sequence[IndType],
        *arg,
        **kwargs,
    ) -> Sequence[IndType]:
        raise NotImplementedError("Expected to be implemented in the subclasses")
