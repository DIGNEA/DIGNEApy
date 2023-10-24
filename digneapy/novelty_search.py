#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   novelty_search.py
@Time    :   2023/10/24 11:23:08
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import reprlib
import operator
from functools import reduce


class Archive:
    _typecode = "d"

    def __init__(self, descriptors: list = None):
        if descriptors:
            # self._descriptors = list(array(self.typecode, d) for d in descriptors)
            self._descriptors = list(d for d in descriptors)
        else:
            self._descriptors = []

    @property
    def descriptors(self):
        return self._descriptors

    def __iter__(self):
        return iter(self._descriptors)

    def __repr__(self):
        components = []
        for d in self._descriptors:
            comp = reprlib.repr(d)
            comp = comp[comp.find("[") : -1]
            components.append(comp)
        return f"Archive({components})"

    def __str__(self):
        return str(tuple(self))

    def __array__(self) -> np.ndarray:
        """
        Creates a ndarray with the descriptors
        """
        return np.array(self._descriptors)

    def __bytes__(self):
        return bytes([ord(self._typecode)]) + bytes(self._descriptors)

    def __eq__(self, other):
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def _descriptor_hash(self, d):
        hashes = (hash(x) for x in d)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __hash__(self):
        hashes = (self._descriptor_hash(d) for d in self._descriptors)
        return reduce(lambda a, b: a ^ b, hashes, 0)

    def __bool__(self):
        return len(self) != 0

    def __len__(self):
        return len(self._descriptors)

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self._descriptors[key])
        index = operator.index(key)
        return self._descriptors[index]

    def append(self, d):
        self._descriptors.append(list(d))

    def extend(self, iterable):
        for i in iterable:
            if type(i) == list:
                self._descriptors.append(i)

    def __format__(self, fmt_spec=""):
        coords = self
        outer_fmt = "({})"
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(", ".join(components))

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)


class NoveltySearch:
    def __init__(self, t_a: float = 0.001, t_ss: float = 0.001, k: int = 15):
        self._t_a = t_a
        self._t_ss = t_ss
        self._k = k
        self._archive = Archive()
        self._solution_set = Archive()

    @property
    def archive(self):
        return self._archive

    @property
    def solution_set(self):
        return self._solution_set

    @property
    def k(self):
        return self._k

    @property
    def t_a(self):
        return self._t_a

    @property
    def t_ss(self):
        return self._t_ss

    def __str__(self):
        return f"NS(t_a={self._t_a},t_ss={self._t_ss},k={self._k},len(a)={len(self._archive)},len(ss)={len(self._solution_set)})"

    def __repr__(self) -> str:
        return f"NS(t_a={self._t_a},t_ss={self._t_ss},k={self._k},len(a)={len(self._archive)},len(ss)={len(self._solution_set)})"

    def sparseness(
        self, descriptors, include_desc: bool = False, verbose: bool = False
    ) -> list[float]:
        if len(descriptors) == 0 or any(len(d) == 0 for d in descriptors):
            msg = f"{self.__class__.__name__} tyring to calculate sparseness on an empty descriptor list"
            raise AttributeError(msg)

        if self._k >= len(descriptors):
            msg = f"{self.__class__.__name__} tyring to calculate sparseness with k({self._k}) > len(descriptors)({len(descriptors)})"
            raise AttributeError(msg)
        # We need to concatentate the archive to the given descriptors
        # and to set k+1 because it predicts n[0] == self descriptor
        _descriptors_arr = (
            np.vstack([self._archive.descriptors, descriptors], dtype=float)
            if len(self._archive)
            else descriptors
        )
        # _descriptors_arr = np.array(
        #     np.concatenate((self._archive.descriptors, descriptors), axis=0),
        #     dtype=float,
        # )

        neighbourhood = NearestNeighbors(n_neighbors=self._k + 1, algorithm="ball_tree")
        neighbourhood.fit(_descriptors_arr)
        sparseness = []
        for descriptor in _descriptors_arr:
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = (1.0 / self._k) * sum(dist)
            sparseness.append(s)
            if include_desc and s >= self._t_a:
                self._archive.append(descriptor)

        return sparseness

    def sparseness_solution_set(self, descriptors):
        if len(descriptors) == 0 or any(len(d) == 0 for d in descriptors):
            msg = f"{self.__class__.__name__} tyring to calculate sparseness_solution_set on an empty descriptor list"
            raise AttributeError(msg)

        if self._k >= len(descriptors):
            msg = f"{self.__class__.__name__} tyring to calculate sparseness_solution_set with k({self._k}) > len(descriptors)({len(descriptors)})"
            raise AttributeError(msg)

        _descriptors_arr = (
            np.vstack([self._solution_set.descriptors, descriptors], dtype=float)
            if len(self._solution_set)
            else descriptors
        )

        neighbourhood = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
        neighbourhood.fit(_descriptors_arr)
        sparseness = []
        for descriptor in _descriptors_arr:
            dist, ind = neighbourhood.kneighbors([descriptor])
            dist, ind = dist[0][1:], ind[0][1:]
            s = (1.0 / self._k) * sum(dist)
            sparseness.append(s)
            if s >= self._t_ss:
                self._solution_set.append(descriptor)
        return s
