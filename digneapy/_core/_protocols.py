#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _protocols.py
@Time    :   2025/04/03 12:05:50
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from abc import abstractmethod
from typing import Optional, Protocol

import numpy as np
from numpy.random import Generator, SeedSequence

from ._types import IndType


class RandGen(Protocol):
    """Protocol to type check all operators have Random Generator attribute (_rng)"""

    _rng: Generator
    _seed: int | None
    _seed_sequence: SeedSequence

    def initialize_rng(self, seed: Optional[int] = None):
        self._seed = seed
        self._seed_sequence = np.random.SeedSequence(self._seed)
        self._rng = np.random.default_rng(self._seed_sequence)


class Transformer(Protocol):
    """Protocol to type check all the transformer types in digneapy.
    A Transformer is any callable type that receives a sequence and transforms it
    to other sequence.
    """

    _name: str

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def __call__(self, x: np.ndarray | list[IndType]) -> np.ndarray: ...

    def train(self, x: np.ndarray | list[IndType]):
        raise NotImplementedError("train method not implemented in Transformer")

    def predict(self, x: np.ndarray | list[IndType]) -> np.ndarray:
        return self.__call__(x)

    def save(self):
        raise NotImplementedError("save method not implemented in Transformer")
