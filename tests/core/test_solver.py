#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_solver.py
@Time    :   2024/06/18 11:45:55
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy.core import Problem, Solution, Solver


class SampleSolver(Solver):
    def __call__(self, problem: Problem, *args, **kwargs) -> list[Solution]:
        return super().__call__(problem, args, kwargs)


@pytest.fixture
def sample_solver():
    return SampleSolver()


def test_raises_not_implemented_solver(sample_solver):
    with pytest.raises(NotImplementedError):
        sample_solver(None)
