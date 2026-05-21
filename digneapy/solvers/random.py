#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   random.py
@Time    :   2026/05/20 15:16:56
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import warnings
from operator import attrgetter
from typing import Optional

import numpy as np

from digneapy._core import Problem, Solution


def random_solver(
    problem: Problem,
    n_solutions: int = 10,
    dtype: type[np.int32 | np.float64] = np.float64,
    seed: Optional[int | np.random.SeedSequence] = None,
) -> list[Solution]:
    if not isinstance(problem, Problem):
        raise TypeError(
            f"Expected a problem in random_solver. Got: {problem.__class__.__name__}"
        )
    if type(n_solutions) is not int or n_solutions < 0:
        warnings.warn(
            "Expected n_solutions to be a positive integer. Got: {type(n_solutions)} = {n_solutions}. Falling back to n_solutions = 10.",
            RuntimeWarning,
        )
        n_solutions = 10
    if dtype not in (np.int32, np.float64):
        warnings.warn(
            f"Expected dtype to be either int or float. Got: {dtype}. Falling back to int.",
            UserWarning,
        )
    lbs, ubs = problem.lbs, problem.ubs
    dimension = problem.dimension
    rng = np.random.default_rng(seed=seed)
    if dtype is np.int32:
        variables = rng.integers(
            low=lbs, high=ubs, size=(n_solutions, dimension), dtype=dtype
        )
    else:
        variables = rng.uniform(low=lbs, high=ubs, size=(n_solutions, dimension))
    objs = [problem.evaluate(v) for v in variables]
    return sorted(
        [
            Solution(
                variables=variables[i],
                objectives=objs[i],
                fitness=objs[i][0],
                dtype=dtype,
                otype=np.float64,
            )
            for i in range(n_solutions)
        ],
        key=attrgetter("fitness"),
    )
