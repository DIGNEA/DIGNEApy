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
from abc import ABC

from collections.abc import Sequence


class Transformer(ABC):
    def __init__(self, name: str):
        self._name = name

    def train(self, X: Sequence[float]):
        raise NotImplementedError("train method not implemented in Transformer")

    def predict(self, X: Sequence[float]):
        raise NotImplementedError("predict method not implemented in Transformer")

    def __call__(self, X: Sequence[float]):
        raise NotImplementedError("__call__ method not implemented in Transformer")

    def save(self):
        raise NotImplementedError("save method not implemented in Transformer")
