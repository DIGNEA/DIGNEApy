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
    """
    Class representing a solution in a genetic algorithm.
    It contains the variables, objectives, constraints, and fitness of the solution.
    """

    def __init__(
        self,
        variables: Optional[Iterable] = [],
        objectives: Optional[Iterable] = [],
        constraints: Optional[Iterable] = [],
        fitness: np.float64 = np.float64(0.0),
        dtype=np.uint32,
        otype=np.float64,
    ):
        """Creates a new solution object.
        The variables is a numpy array of the solution's genes.
        The objectives and constraints are numpy arrays of the solution's objectives and constraints.
        The fitness is a float representing the solution's fitness value.

        Args:
            variables (Optional[Iterable], optional): Tuple or any other iterable with the variables/variables. Defaults to None.
            objectives (Optional[Iterable], optional): Tuple or any other iterable with the objectives values. Defaults to None.
            constraints (Optional[Iterable], optional): Tuple or any other iterable with the constraint values. Defaults to None.
            fitness (float, optional): Fitness of the solution. Defaults to 0.0.
        """

        self._otype = otype
        self._dtype = dtype
        self.variables = np.asarray(variables, dtype=self.dtype)
        self.objectives = np.array(objectives, dtype=self.otype)
        self.constraints = np.array(constraints, dtype=self.otype)
        self.fitness = otype(fitness)

    @property
    def dtype(self):
        return self._dtype

    @property
    def otype(self):
        return self._otype

    def clone(self) -> Self:
        """Returns a deep copy of the solution. It is more efficient than using the copy module.

        Returns:
            Self: Solution object
        """
        return Solution(
            variables=list(self.variables),
            objectives=list(self.objectives),
            constraints=list(self.constraints),
            fitness=self.fitness,
            otype=self.otype,
        )

    def clone_with(self, **overrides):
        """Clones an Instance with overriden attributes

        Returns:
            Instance
        """
        new_object = self.clone()
        for key, value in overrides.items():
            setattr(new_object, key, value)
        return new_object

    def __str__(self) -> str:
        return f"Solution(dim={len(self.variables)},f={self.fitness},objs={self.objectives},const={self.constraints})"

    def __repr__(self) -> str:
        return f"Solution<dim={len(self.variables)},f={self.fitness},objs={self.objectives},const={self.constraints}>"

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

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
            return cls(self.variables[key])
        index = operator.index(key)
        return self.variables[index]

    def __setitem__(self, key, value):
        self.variables[key] = value
