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
from collections.abc import Iterable, Sequence
from typing import Optional, Self

import numpy as np


class Solution:
    """
    Class representing a solution in a genetic algorithm.

    It contains the variables, objectives, constraints, and fitness of the solution.
    """

    __slots__ = (
        "_variables",
        "_objectives",
        "_constraints",
        "_fitness",
        "_dtype",
        "_otype",
    )

    def __init__(
        self,
        variables: Optional[Iterable] = [],
        objectives: Optional[Iterable] = [],
        constraints: Optional[Iterable] = [],
        fitness: float | np.float64 = np.float64(0.0),
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
        self._variables = np.asarray(variables, dtype=self.dtype)
        self._objectives = np.asarray(objectives, dtype=self.otype)
        self._constraints = np.asarray(constraints, dtype=self.otype)
        try:
            if fitness is None:
                self._fitness = np.float64(0)
            else:
                self._fitness = np.float64(fitness)
        except ValueError:
            raise ValueError(
                "Fitness must be convertible to float in Solution.__init__()"
            )

    @property
    def dtype(self):
        return self._dtype

    @property
    def otype(self):
        return self._otype

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, new_variables: np.ndarray):
        if len(new_variables) != len(self._variables):
            raise ValueError(
                "Updating the variables of an Solution object with a different number of values. "
                f"Solution have {len(self._variables)} "
                f"variables and the new_variables sequence have {len(new_variables)}"
            )
        self._variables = np.asarray(new_variables)

    @property
    def fitness(self) -> np.float64:
        return self._fitness

    @fitness.setter
    def fitness(self, f: np.float64):
        try:
            self._fitness = np.float64(f)
            print(self._fitness)
        except ValueError:
            # if f != 0.0 and not float(f):
            msg = f"The fitness value {f} is not a float in fitness setter of class Solution."
            raise ValueError(msg)

    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, new_objectives: np.ndarray):
        if len(new_objectives) != len(self._objectives):
            raise ValueError(
                "Updating the objectives of an Solution object with a different number of values. "
                f"Solution have {len(self._objectives)} "
                f"objectives and the new_objectives sequence have {len(new_objectives)}"
            )
        self._objectives = np.asarray(new_objectives)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, new_constraints: np.ndarray):
        if len(new_constraints) != len(self._constraints):
            raise ValueError(
                "Updating the constraints of an Solution object with a different number of values. "
                f"Solution have {len(self._constraints)} "
                f"constraints and the new_constraints sequence have {len(new_constraints)}"
            )
        self._constraints = np.asarray(new_constraints)

    def clone(self) -> Self:
        """Returns a deep copy of the solution. It is more efficient than using the copy module.

        Returns:
            Self: Solution object
        """
        return type(self)(
            variables=list(self._variables),
            objectives=list(self._objectives),
            constraints=list(self._constraints),
            fitness=self._fitness,
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

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, Solution):
            raise TypeError(
                "Other of type {other.__class__.__name__} can not be compared with a Solution."
            )
        else:
            try:
                return all(a == b for a, b in zip(self, other, strict=True))
            except ValueError:
                return False

    def __gt__(self, other: Self) -> np.bool:
        if not isinstance(other, Solution):
            raise TypeError(
                "Other of type {other.__class__.__name__} can not be compared with a Solution."
            )

        return self.fitness > other.fitness

    def __getitem__(self, key: int | slice) -> Sequence | np.ndarray:
        """Accessor to variables of the Solution

        Args:
            key (int | slice): index or slice to access a subset of variables

        Returns:
            Sequence or np.ndarray: If accessed with a slice the subset of variables
                are returned. Otherwise, it returns a single scalar at variables[key].
        """
        if not isinstance(key, (int, slice)):
            raise TypeError(
                f"Solution cannot be indexed with type: {type(key)}. Use slice or int."
            )
        if isinstance(key, slice):
            return self._variables[key]
        else:
            index = operator.index(key)
            return self.variables[index]

    def __setitem__(self, key: int | slice, value):
        """Setter to variables of the Solution

        Args:
            key (int | slice): index or slice to access a subset of variables
            value: Value to set in the variables

        """
        if not isinstance(key, (int, np.integer, np.unsignedinteger, slice)):
            raise TypeError(
                f"Solution cannot be update via __setitem__ with type: {type(key)}. Use slice or int."
            )

        if isinstance(key, slice):
            # Compute how many positions this slice actually targets
            start, stop, step = key.indices(len(self.variables))
            target_count = len(range(start, stop, step))
            # Accept any sequence; reject bare scalars for slice assignment
            try:
                expected_len = len(value)
            except TypeError:
                raise TypeError(
                    f"[Solution] slice assignment requires a sequence, got scalar {value!r}. "
                    f"Expected {target_count} value(s)."
                )

            if expected_len != target_count:
                raise ValueError(
                    f"[Solution] slice targets {target_count} element(s) but value has "
                    f"{expected_len} element(s)."
                )
        else:
            index = operator.index(key)
            try:
                if len(value) != 1:
                    raise ValueError(
                        f"[Solution] index {index!r} targets 1 element but value has "
                        f"{len(value)} element(s). Use a slice to assign multiple values."
                    )
            except TypeError:
                pass  # len() failed which means that we have a scalar value

        self._variables[key] = value
