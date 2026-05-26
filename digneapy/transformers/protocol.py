#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   protocol.py
@Time    :   2026/05/18 10:57:45
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod

import numpy as np

from .._core._types import IndType


class Transformer(ABC):
    """Base class to type check all the transformer types in digneapy.
    A Transformer is any callable type that receives a sequence and transforms it
    to other sequence.
    """

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
