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
    It contains the chromosome, objectives, constraints, and fitness of the solution.
    """

    def __init__(
        self,
        chromosome: Optional[Iterable] = [],
        objectives: Optional[Iterable] = [],
        constraints: Optional[Iterable] = [],
        fitness: float = 0.0,
        otype=np.float64,
    ):
        """Creates a new solution object.
        The chromosome is a numpy array of the solution's genes.
        The objectives and constraints are numpy arrays of the solution's objectives and constraints.
        The fitness is a float representing the solution's fitness value.

        Args:
            chromosome (Optional[Iterable], optional): Tuple or any other iterable with the chromosome/variables. Defaults to None.
            objectives (Optional[Iterable], optional): Tuple or any other iterable with the objectives values. Defaults to None.
            constraints (Optional[Iterable], optional): Tuple or any other iterable with the constraint values. Defaults to None.
            fitness (float, optional): Fitness of the solution. Defaults to 0.0.
        """

        self.otype = otype
        self.chromosome = np.asarray(chromosome)
        self.objectives = np.array(objectives, dtype=self.otype)
        self.constraints = np.array(constraints, dtype=self.otype)
        self.fitness = otype(fitness)

    def clone(self) -> Self:
        """Returns a deep copy of the solution. It is more efficient than using the copy module.

        Returns:
            Self: Solution object
        """
        return Solution(
            chromosome=list(self.chromosome),
            objectives=list(self.objectives),
            constraints=list(self.constraints),
            fitness=self.fitness,
            otype=self.otype,
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
