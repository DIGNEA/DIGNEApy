#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   instance.py
@Time    :   2024/06/07 14:09:43
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


import copy
import operator
import reprlib
from collections.abc import Iterable
from functools import reduce
from typing import Tuple, Optional


class Instance:
    def __init__(
        self,
        variables: Optional[Iterable] = None,
        fitness: float = 0.0,
        p: float = 0.0,
        s: float = 0.0,
    ):
        if variables is not None:
            self._variables = list(variables)
        else:
            self._variables = []
        try:
            fitness = float(fitness)
            p = float(p)
            s = float(s)
        except ValueError:
            raise AttributeError(
                "The fitness, p and s parameters must be convertible to float"
            )
        self._fitness = fitness
        self._p = p
        self._s = s
        self._portfolio_m: Tuple = tuple()
        self._features: Tuple = tuple()

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, performance: float):
        try:
            performance = float(performance)
        except ValueError:
            # if performance != 0.0 and not float(performance):
            msg = f"The performance value {performance} is not a float in 'p' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._p = performance

    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, novelty: float):
        try:
            novelty = float(novelty)
        except ValueError:
            # if novelty != 0.0 and not float(novelty):
            msg = f"The novelty value {novelty} is not a float in 's' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._s = novelty

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, f: float):
        try:
            f = float(f)
        except ValueError:
            # if f != 0.0 and not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class {self.__class__.__name__}"
            raise AttributeError(msg)

        self._fitness = float(f)

    @property
    def features(self) -> tuple:
        return self._features

    @features.setter
    def features(self, f: tuple):
        self._features = f

    @property
    def portfolio_scores(self):
        return self._portfolio_m

    @portfolio_scores.setter
    def portfolio_scores(self, p: tuple):
        self._portfolio_m = copy.deepcopy(p)

    def __repr__(self):
        return f"Instance<f={self._fitness},p={self._p},s={self._s},vars={len(self._variables)},features={len(self._features)},performance={len(self._portfolio_m)}>"

    def __str__(self):
        features = reprlib.repr(self._features)
        performance = reprlib.repr(self._portfolio_m)
        performance = performance[performance.find("(") : performance.rfind(")") + 1]
        return f"Instance(f={self._fitness},p={self._p},s={self._s},features={features},performance={performance})"

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._variables[key])
        index = operator.index(key)
        return self._variables[index]

    def __setitem__(self, key, value):
        self._variables[key] = value

    def __eq__(self, other):
        if isinstance(other, Instance):
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False
        else:
            return NotImplemented

    def __gt__(self, other):
        if not isinstance(other, Instance):
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            print(msg)
            return NotImplemented
        return self.fitness > other.fitness

    def __ge__(self, other):
        if not isinstance(other, Instance):
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            print(msg)
            return NotImplemented
        return self.fitness >= other.fitness

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
        msg = f"Instance(f={self._fitness},p={format(self._p, fmt_spec)}, s={format(self._s, fmt_spec)}, {decriptor})"

        return msg
