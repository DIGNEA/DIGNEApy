#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   domain.py
@Time    :   2023/10/24 17:04:31
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""


import reprlib
import math
from functools import reduce


class Instance:
    def __init__(
        self,
        variables: list = None,
        fitness: float = 0.0,
        p: float = 0.0,
        s: float = 0.0,
    ):
        if variables:
            self._variables = list(variables)
        else:
            self._variables = []
        self._fitness = fitness
        self._p = p
        self._s = s
        self._portfolio_m = []
        self._features = []

    @property
    def p(self):
        return self._p

    @property.setter
    def p(self, performance):
        if not float(performance):
            msg = f"The performance value {performance} is not a float in 'p' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._p = performance

    @property
    def s(self):
        return self._s

    @property.setter
    def s(self, novelty):
        if not float(novelty):
            msg = f"The novelty value {novelty} is not a float in 's' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._s = novelty

    @property
    def fitness(self):
        return self._fitness

    @property.setter
    def fitness(self, f):
        if not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._fitness = f

    def __repr__(self):
        return f"Instance(f={self._fitness},p={self._p},s={self._s})"

    def __str__(self):
        features = reprlib.repr(self._features)
        performance = reprlib.repr(self._portfolio_m)
        return f"Instance(f={self._fitness},p={self._p},s={self._s},features={features},performance={performance})"

    def __iter__(self):
        return iter(self._variables)

    def __eq__(self, other):
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __hash__(self):
        hashes = (hash(x) for x in self._variables)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __abs__(self):
        return math.hypot(*self)

    def __bool__(self):
        return bool(abs(self))

    def feature_descriptor(self):
        return self._features

    def performance_descriptor(self):
        return self._portfolio_m
