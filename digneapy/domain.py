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
import operator
import random
from functools import reduce
from typing import TypeVar, Generic, Union, get_args, Iterable, Tuple


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

    def calculate_features(self):
        """Calculates the features of the instance"""
        msg = f"Method not implemented in class Instance. Must be re-defined in subclasses"
        raise NotImplementedError(msg)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, performance):
        if not float(performance):
            msg = f"The performance value {performance} is not a float in 'p' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._p = performance

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, novelty):
        if not float(novelty):
            msg = f"The novelty value {novelty} is not a float in 's' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._s = novelty

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, f):
        if not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._fitness = f

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, f):
        self._features = f

    @property
    def performance(self):
        return self._portfolio_m

    @performance.setter
    def performance(self, p):
        self._portfolio_m = p

    def __repr__(self):
        return f"Instance(f={self._fitness},p={self._p},s={self._s},vars={len(self._variables)},features={len(self._features)},performance={len(self._portfolio_m)})"

    def __str__(self):
        features = reprlib.repr(self._features)
        features = features[features.find("[") : features.find("]") + 1]
        performance = reprlib.repr(self._portfolio_m)
        performance = performance[performance.find("[") : performance.find("]") + 1]
        return f"Instance(f={self._fitness},p={self._p},s={self._s},features={features},performance={performance})"

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __eq__(self, other):
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __hash__(self):
        hashes = (hash(x) for x in self)
        return reduce(operator.or_, hashes, 0)

    def __bool__(self):
        return bool(self._variables)

    def __format__(self, fmt_spec=""):
        if fmt_spec.endswith("p"):
            # We are showing only the performances
            fmt_spec = fmt_spec[:-1]
            components = self._portfolio_m
        else:
            fmt_spec = fmt_spec[:-1]
            components = self._features

        components = (format(c, fmt_spec) for c in components)
        decriptor = "descriptor=({})".format(",".join(components))
        msg = f"Instance(p={format(self._p, fmt_spec)}, s={format(self._s, fmt_spec)}, {decriptor})"

        return msg


T = TypeVar("T", bound=Union[int, float])


class Domain:
    def __init__(
        self, name: str = "", size: int = 0, bounds: Iterable[Tuple] = None, dtype=int
    ):
        self._name = name
        self._dimension = size
        self._bounds = list(bounds)
        self._dtype = dtype

    @property
    def bounds(self):
        return self._bounds

    def create_instance(self) -> Instance:
        """Creates a new instances for the optimisation domain by means of
        drawing random values in the range (l_i, u_i) for i in [0, size]
        """

        variables = [
            self._dtype(random.uniform(l_i, u_i)) for (l_i, u_i) in self._bounds
        ]
        return Instance(variables)

    def __repr__(self):
        return f"Domain<name={self._name},size={self._dimension},bounds={self._bounds},dtype={self._dtype}>"
