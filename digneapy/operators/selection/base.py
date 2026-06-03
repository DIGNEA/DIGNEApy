#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   base.py
@Time    :   2026/05/21 15:16:02
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC
from collections.abc import Sequence
from typing import Optional

import numpy as np

from digneapy.typing import IndType


class Selection(ABC):
    def __init__(self, seed: Optional[int | np.random.SeedSequence] = None):
        self._seed = (
            seed
            if isinstance(seed, (int | np.random.SeedSequence))
            else np.random.SeedSequence()
        )
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        population: Sequence[IndType] | np.ndarray,
        *arg,
        **kwargs,
    ):
        raise NotImplementedError("Expected to be implemented in the subclasses")
