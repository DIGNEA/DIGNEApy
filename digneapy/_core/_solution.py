#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   solution.py
@Time    :   2024/06/07 14:09:54
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import operator
from collections.abc import Iterable
from typing import Optional, Self
import numpy as np


class Solution:
    def __init__(
        self,
        chromosome: Optional[Iterable] = None,
        objectives: Optional[Iterable] = None,
        constraints: Optional[Iterable] = None,
        fitness: float = 0.0,
    ):
        if chromosome is not None:
            self.chromosome = np.asarray(chromosome)
        else:
            self.chromosome = np.empty(0)
        self.objectives = np.array(objectives) if objectives else np.empty(0)
        self.constraints = np.array(constraints) if constraints else np.empty(0)
        self.fitness = fitness

    def clone(self) -> Self:
        return Solution(
            chromosome=list(self.chromosome),
            objectives=list(self.objectives),
            constraints=list(self.constraints),
            fitness=self.fitness,
        )

    def __str__(self) -> str:
        return f"Solution(dim={len(self.chromosome)},f={self.fitness},objs={self.objectives},const={self.constraints})"

    def __repr__(self) -> str:
        return f"Solution<dim={len(self.chromosome)},f={self.fitness},objs={self.objectives},const={self.constraints}>"

    def __len__(self) -> int:
        return len(self.chromosome)

    def __iter__(self):
        return iter(self.chromosome)

    def __bool__(self):
        return len(self) != 0

    def __eq__(self, other) -> bool:
        if isinstance(other, Solution):
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False
        else:
            return NotImplemented

    def __gt__(self, other):
        if not isinstance(other, Solution):
            msg = f"Other of type {other.__class__.__name__} can not be compared with with {self.__class__.__name__}"
            print(msg)
            return NotImplemented
        return self.fitness > other.fitness

    def __getitem__(self, key):
        if isinstance(key, slice):
            cls = type(self)  # To facilitate subclassing
            return cls(self.chromosome[key])
        index = operator.index(key)
        return self.chromosome[index]

    def __setitem__(self, key, value):
        self.chromosome[key] = value
