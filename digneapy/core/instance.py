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

import json
import operator
import reprlib
from collections.abc import Iterable
from functools import reduce
from typing import Optional

import numpy as np


class Instance:
    __slots__ = ("_vars", "_fit", "_p", "_s", "_features", "_desc", "_pscores")

    def __init__(
        self,
        variables: Optional[Iterable] = None,
        fitness: float = 0.0,
        p: float = 0.0,
        s: float = 0.0,
        features: Optional[tuple[float]] = None,
        descriptor: Optional[tuple[float]] = None,
        portfolio_scores: Optional[tuple[float]] = None,
    ):
        try:
            fitness = float(fitness)
            p = float(p)
            s = float(s)
        except ValueError:
            raise ValueError(
                "The fitness, p and s parameters must be convertible to float"
            )
        self._vars = np.asarray(variables) if variables is not None else np.empty(0)
        self._fit = fitness
        self._p = p
        self._s = s
        self._features = tuple(features) if features else tuple()
        self._pscores = tuple(portfolio_scores) if portfolio_scores else tuple()
        self._desc = tuple(descriptor) if descriptor else tuple()

    @property
    def variables(self):
        return self._vars

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
            raise ValueError(msg)
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
            raise ValueError(msg)
        self._s = novelty

    @property
    def fitness(self) -> float:
        return self._fit

    @fitness.setter
    def fitness(self, f: float):
        try:
            f = float(f)
        except ValueError:
            # if f != 0.0 and not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class {self.__class__.__name__}"
            raise ValueError(msg)

        self._fit = f

    @property
    def features(self) -> tuple:
        return self._features

    @features.setter
    def features(self, features: tuple):
        self._features = features

    @property
    def descriptor(self) -> tuple:
        return self._desc

    @descriptor.setter
    def descriptor(self, desc: tuple):
        self._desc = desc

    @property
    def portfolio_scores(self):
        return self._pscores

    @portfolio_scores.setter
    def portfolio_scores(self, p: tuple):
        self._pscores = tuple(p)

    def __repr__(self):
        return f"Instance<f={self.fitness},p={self.p},s={self.s},vars={len(self._vars)},features={len(self.features)},descriptor={len(self.descriptor)},performance={len(self.portfolio_scores)}>"

    def __str__(self):
        descriptor = reprlib.repr(self.descriptor)
        performance = reprlib.repr(self.portfolio_scores)
        performance = performance[performance.find("(") : performance.rfind(")") + 1]
        return f"Instance(f={self.fitness},p={self.p},s={self.s},features={len(self.features)},descriptor={descriptor},performance={performance})"

    def __iter__(self):
        return iter(self._vars)

    def __len__(self):
        return len(self._vars)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._vars[key])
        index = operator.index(key)
        return self._vars[index]

    def __setitem__(self, key, value):
        self._vars[key] = value

    def __eq__(self, other):
        if isinstance(other, Instance):
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False
        else:
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            raise NotImplementedError(msg)

    def __gt__(self, other):
        if not isinstance(other, Instance):
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            raise NotImplementedError(msg)

        return self.fitness > other.fitness

    def __ge__(self, other):
        if not isinstance(other, Instance):
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            raise NotImplementedError(msg)

        return self.fitness >= other.fitness

    def __hash__(self):
        hashes = (hash(x) for x in self)
        return reduce(operator.or_, hashes, 0)

    def __bool__(self):
        return self._vars.size != 0

    def __format__(self, fmt_spec=""):
        if fmt_spec.endswith("p"):
            # We are showing only the performances
            fmt_spec = fmt_spec[:-1]
            components = self.portfolio_scores
        else:
            fmt_spec = fmt_spec[:-1]
            components = self.descriptor

        components = (format(c, fmt_spec) for c in components)
        decriptor = "descriptor=({})".format(",".join(components))
        msg = f"Instance(f={self.fitness},p={format(self.p, fmt_spec)}, s={format(self.s, fmt_spec)}, {decriptor})"

        return msg

    def to_json(self):
        data = {
            "fitness": self.fitness,
            "s": self.s,
            "p": self.p,
            "portfolio": self.portfolio_scores,
            "variables": self._vars.tolist(),
            "features": self.features,
            "descriptor": self.descriptor,
        }
        return json.dumps(data, sort_keys=True, indent=4)
