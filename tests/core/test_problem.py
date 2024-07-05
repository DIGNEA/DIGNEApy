#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_problem.py
@Time    :   2024/06/18 11:38:27
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from typing import Sequence, Tuple

import pytest

from digneapy.core import Problem
from digneapy.core.solution import Solution


class Sample(Problem):
    def __init__(self):
        dimension = 100
        bounds = [(0, 10) for _ in range(dimension)]
        super().__init__(dimension, bounds)

    def create_solution(self) -> Solution:
        return super().create_solution()

    def evaluate(self, individual: Sequence | Solution) -> Tuple[float]:
        return super().evaluate(individual)

    def __call__(self, individual: Sequence) -> Tuple[float]:
        return self.evaluate(individual)

    def to_file(self, filename: str):
        return super().to_file(filename)


@pytest.fixture
def sample_problem():
    sample = Sample()
    return sample


def test_problem_methods(sample_problem):
    assert sample_problem._name == "DefaultProblem"
    assert sample_problem.dimension == 100

    expected_bounds = list((0, 10) for _ in range(100))
    assert sample_problem.bounds == expected_bounds
    assert all(sample_problem.get_bounds_at(i) == (0.0, 10.0) for i in range(100))

    with pytest.raises(NotImplementedError):
        sample_problem.create_solution()

    with pytest.raises(NotImplementedError):
        sample_problem.evaluate(list())

    with pytest.raises(NotImplementedError):
        sample_problem.to_file(filename="")

    with pytest.raises(NotImplementedError):
        Sample.from_file("")

    with pytest.raises(ValueError):
        sample_problem.get_bounds_at(-1)

    with pytest.raises(ValueError):
        sample_problem.get_bounds_at(1000)
