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
from typing import Generator

from .._core._instance import Instance
from .._core._solver import Solver


def cast_to_instances(
    genotypes, descriptors, fitness, portfolio_scores, diversity_scores, bias_score
) -> list[Instance]:
    """Creates objects of type Instance from a collection of np.ndarray

    Args:
        genotypes (_type_): Genotypes of the instances
        descriptors (_type_): Descriptors of the instances
        fitness (_type_): Fitness values of the instances
        portfolio_scores (_type_): Scores of the instances
        diversity_scores (_type_): Diversity scores of the instances
        bias_score (_type_): Performance bias scores of the instances

    Raises:
        RuntimeError: If the len() of any list differs from the rest

    Returns:
        list[Instance]: List of Instance objects ready to be inserted in the archives
    """
    expected = len(genotypes)
    if any(
        len(l) != expected
        for l in (
            genotypes,
            descriptors,
            fitness,
            portfolio_scores,
            diversity_scores,
            fitness,
            bias_score,
        )
    ):
        raise RuntimeError("Length mismatch")
    return [
        Instance(
            variables=genotypes[i],
            fitness=fitness[i],
            descriptor=descriptors[i],
            p=bias_score[i],
            portfolio_scores=portfolio_scores[i],
            s=diversity_scores[i],
        )
        for i in range(expected)
    ]


def extract_solvers_name(portfolio: Sequence[Solver]) -> Generator[str]:
    """Simple generator to extract the names of the solvers in the portfolio

    Args:
        portfolio (Sequence[Solver]): Sequence of solvers used to evaluate the instances

    Yields:
        Generator[str]: Generator of strings
    """
    for solver in portfolio:
        yield solver.__name__
