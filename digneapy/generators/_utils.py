#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/05/28 14:03:49
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Any, Generator, Self

import numpy as np

from .._core._instance import Instance
from .._core._solver import Solver


class InstanceBuilder:
    """Class to create Instances using the Builder Pattern
    - Call add_component with the key and value to include in the new Instance
    - Finally call build() to generate a new Instance
    - Calling build() flushes the given components and gets the builder object ready to start again
    - Expected keys are: variables, fitness, descriptor, portfolio_scores, p, s.
    """

    def __init__(self):
        self._components: dict[str, Any] = {}

    def add_component(self, key: str, value: Any) -> Self:
        self._components[key] = value
        return self

    def build(self) -> Instance:
        if "variables" not in self._components:
            raise ValueError("Cannot create an Instance without variables.")
        instance = Instance(variables=self._components["variables"])
        for key, value in self._components.items():
            if key == "variables":
                continue
            setattr(instance, key, value)

        self._components.clear()
        return instance


def cast_to_instances(
    genotypes: np.ndarray,
    descriptors: np.ndarray,
    fitness: np.ndarray,
    portfolio_scores: np.ndarray,
    diversity_scores: np.ndarray,
    bias_score: np.ndarray,
) -> list[Instance]:
    """Creates objects of type Instance from a collection of np.ndarray

    Args:
        genotypes (np.ndarray): Genotypes of the instances
        descriptors (np.ndarray): Descriptors of the instances
        fitness (np.ndarray): Fitness values of the instances
        portfolio_scores (np.ndarray): Scores of the instances
        diversity_scores (np.ndarray): Diversity scores of the instances
        bias_score (np.ndarray): Performance bias scores of the instances

    Raises:
        RuntimeError: If the len() of any np.ndarray differs from the rest

    Returns:
        list[Instance]: List of Instance objects ready to be inserted in the archives
    """
    expected = len(genotypes)
    if any(
        len(component) != expected
        for component in (
            genotypes,
            descriptors,
            fitness,
            portfolio_scores,
            diversity_scores,
            fitness,
            bias_score,
        )
    ):
        raise RuntimeError("Length mismatch of components in cast_to_instances")
    return [
        Instance(
            variables=genotypes[i],
            fitness=fitness[i],
            descriptor=descriptors[i],
            performance_bias=bias_score[i],
            portfolio_scores=portfolio_scores[i],
            novelty=diversity_scores[i],
        )
        for i in range(expected)
    ]


def extract_solvers_name(portfolio: Sequence[Solver]) -> Generator[str, None, None]:
    """Simple generator to extract the names of the solvers in the portfolio

    Args:
        portfolio (Sequence[Solver]): Sequence of solvers used to evaluate the instances

    Yields:
        Generator[str]: Generator of strings
    """
    for solver in portfolio:
        yield solver.__name__
