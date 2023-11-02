#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   domain.py
@Time    :   2023/10/30 12:48:56
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""


import reprlib
import operator
from functools import reduce
from typing import Iterable, Tuple, Callable
import copy


class OptProblem:
    def __init__(self, name: str = "DefaultOptProblem", *args, **kwargs):
        self._name = name

    def evaluate(self, individual: Iterable) -> Tuple[float]:
        msg = "evaluate method not implemented in OptProblem"
        raise NotImplementedError(msg)


"""Solver is any callable type that receives a OptProblem 
as its argument and returns a tuple with the solution found"""
Solver = Callable[[OptProblem], Tuple]


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
        self._portfolio_m = list()
        self._features = list()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, performance):
        if performance != 0.0 and not float(performance):
            msg = f"The performance value {performance} is not a float in 'p' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._p = performance

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, novelty):
        if novelty != 0.0 and not float(novelty):
            msg = f"The novelty value {novelty} is not a float in 's' setter of class {self.__class__.__name__}"
            raise AttributeError(msg)
        self._s = novelty

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, f):
        if f != 0.0 and not float(f):
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
    def portfolio_scores(self):
        return self._portfolio_m

    @portfolio_scores.setter
    def portfolio_scores(self, p):
        self._portfolio_m = copy.deepcopy(p)

    def __repr__(self):
        return f"Instance<f={self._fitness},p={self._p},s={self._s},vars={len(self._variables)},features={len(self._features)},performance={len(self._portfolio_m)}>"

    def __str__(self):
        features = reprlib.repr(self._features)
        performance = reprlib.repr(self._portfolio_m)
        performance = performance[performance.find("[") : performance.rfind("]") + 1]
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
        msg = f"Instance(f={self._fitness},p={format(self._p, fmt_spec)}, s={format(self._s, fmt_spec)}, {decriptor})"

        return msg


class Domain:
    def __init__(
        self,
        name: str = "Domain",
        dimension: int = 0,
        bounds: Iterable[Tuple] = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.dimension = dimension
        self.bounds = bounds if bounds else [(0.0, 0.0)]

    def generate_instance(self) -> Instance:
        msg = "generate_instances is not implemented in Domain class."
        raise NotImplementedError(msg)

    def extract_features(self, instance: Instance) -> list[float]:
        msg = "extract_features is not implemented in Domain class."
        raise NotImplementedError(msg)

    def from_instance(self, instance: Instance) -> OptProblem:
        msg = "from_instance is not implemented in Domain class."
        raise NotImplementedError(msg)

    def __len__(self):
        return self.dimension

    def lower_i(self, i):
        if i < 0 or i > len(self.bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self.bounds)}]"
            raise AttributeError(msg)
        return self.bounds[i][0]

    def upper_i(self, i):
        if i < 0 or i > len(self.bounds):
            msg = f"index {i} is out of bounds. Valid values are [0-{len(self.bounds)}]"
            raise AttributeError(msg)
        return self.bounds[i][1]
