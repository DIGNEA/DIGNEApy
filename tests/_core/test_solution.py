#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_solution.py
@Time    :   2024/06/18 11:38:51
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy import Solution


def test_solution_attrs():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    assert len(solution) == dimension
    assert len(solution.objectives) == 2
    assert len(solution.constraints) == 2
    assert solution.fitness == np.float64(100.0)
    assert np.array_equal(solution.variables, np.arange(dimension))


def test_solution_raises_init():
    with pytest.raises(ValueError):
        dimension = 10
        _ = Solution(
            variables=list(range(dimension)),
            objectives=(0.0, 1.0),
            constraints=(
                0.0,
                0.0,
            ),
            fitness="NonValidFitness",
        )


def test_solution_none_fitness_to_zero():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=None,
    )
    assert_equal(solution.fitness, np.float64(0))


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("fitness", 10),
        ("objectives", (2.0, 5.0)),
        ("constraints", (0.5, 0.0)),
        ("variables", np.zeros(10)),
    ],
)
def test_solution_setters_work_as_expected(keyword, value):
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    previous_value = getattr(solution, keyword)
    setattr(solution, keyword, value)
    current_value = getattr(solution, keyword)
    assert not np.array_equal(previous_value, current_value)
    assert_equal(value, current_value)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("fitness", "NonValidFitness"),
        ("objectives", (2.0,)),
        ("constraints", (0.5,)),
        ("variables", np.zeros(1)),
    ],
)
def test_solution_setters_raises_if_wrong(keyword, value):
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(ValueError):
        setattr(solution, keyword, value)


def test_solution_can_be_cloned():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    cloned = solution.clone()
    assert cloned == solution
    assert cloned is not solution
    assert len(cloned) == len(solution)

    assert len(cloned.objectives) == len(solution.objectives)
    assert_equal(cloned.objectives, solution.objectives)

    assert len(cloned.constraints) == len(solution.constraints)
    assert_equal(cloned.constraints, solution.constraints)

    assert cloned.fitness == solution.fitness
    assert_equal(cloned.variables, solution.variables)


@pytest.mark.parametrize(
    "keyword, value",
    [("fitness", 10), ("objectives", (2.0, 3.0)), ("constraints", (0.5, 0.0))],
)
def test_solution_can_be_cloned_with(keyword, value):
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    cloned = solution.clone_with(**{keyword: value})
    assert cloned == solution  # same length and variables
    assert cloned is not solution
    assert_equal(getattr(cloned, keyword), value)


def test_solution_can_be_iter():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    assert_equal(list(solution), np.arange(dimension))


def test_solution_can_be_compared_true():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    cloned = solution.clone()
    assert cloned == solution


def test_solution_can_be_compared_false_same_dim():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    other = Solution(
        variables=list(range(1, dimension + 1)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    assert len(other) == len(solution)
    assert other != solution


def test_solution_can_be_compared_false_diff_dim():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    other = Solution(
        variables=list(range(dimension // 2)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    assert len(other) == len(solution) // 2
    assert other != solution


def test_solution_can_be_compared_by_fitness():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    other = solution.clone()
    # Same dimension and variables
    assert other == solution
    # Now other has a greater fitness
    other.fitness = solution.fitness + 1.0
    assert other > solution
    # Now other has a smaller fitness
    other.fitness = solution.fitness - 1.0
    assert other < solution


def test_solution_cannot_be_comp_with_other_types():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(TypeError):
        solution == np.ndarray

    with pytest.raises(TypeError):
        solution > 1.0


def test_solution_can_be_accessed():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    assert solution[0] == 0
    assert solution[-1] == 9
    assert_equal(solution[2:4], [2, 3])


@pytest.mark.parametrize("index", (None, 2.5, "abc"))
def test_solution_cannot_be_accessed_with_others(index):
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(TypeError):
        _ = solution[index]


def test_solution_can_be_accessed_and_raise():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(IndexError):
        _ = solution[1000]


def test_solution_can_be_updated_setitem():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    expected = 100
    solution[0] = expected
    assert solution[0] == expected

    solution[-1] = expected
    assert solution[-1] == expected

    expected_slice = [100, 200, 300]
    solution[2:5] = expected_slice
    assert_equal(solution[2:5], expected_slice)


@pytest.mark.parametrize("index", (None, 2.5, "abc"))
def test_solution_cannot_be_updated_with_others(index):
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(TypeError):
        solution[index] = 1


def test_solution_can_be_updated_and_raise():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(IndexError):
        solution[1000] = 100


def test_solution_setitem_raise_slice_mismatch():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(ValueError):
        solution[1:3] = [10, 20, 30]


def test_solution_setitem_raise_slice_and_scalar_value():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(TypeError):
        solution[1:3] = 100


def test_solution_setitem_raise_index_and_slice_value():
    dimension = 10
    solution = Solution(
        variables=list(range(dimension)),
        objectives=(0.0, 1.0),
        constraints=(
            0.0,
            0.0,
        ),
        fitness=100.0,
    )
    with pytest.raises(ValueError):
        solution[1] = [100, 200, 300]
