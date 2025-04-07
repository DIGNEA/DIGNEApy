#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   types.py
@Time    :   2025/04/03 12:05:50
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from typing import Protocol, Optional
import numpy as np


class RNG(Protocol):
    """Protocol to type check all operators have _rng of instances types in digneapy"""

    _rng: np.random.Generator
    _seed: int

    def initialize_rng(self, seed: Optional[int] = None):
        self._seed = seed if seed else 42
        self._rng = np.random.default_rng(self._seed)
