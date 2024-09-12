#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   transformers.py
@Time    :   2023/11/15 08:51:42
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import numpy.typing as npt


class SupportsTransform(Protocol):
    """Protocol to type check all the transformer types in digneapy.
    A Transformer is any callable type that receives a sequence and transforms it
    to other sequence.
    """

    def __call__(self, X: npt.NDArray) -> np.ndarray: ...


class Transformer(ABC, SupportsTransform):
    def __init__(self, name: str):
        self._name = name

    def train(self, X: npt.NDArray):
        raise NotImplementedError("train method not implemented in Transformer")

    def predict(self, X: npt.NDArray) -> np.ndarray:
        raise NotImplementedError("predict method not implemented in Transformer")

    @abstractmethod
    def __call__(self, X: npt.NDArray) -> np.ndarray:
        raise NotImplementedError("__call__ method not implemented in Transformer")

    def save(self):
        raise NotImplementedError("save method not implemented in Transformer")
