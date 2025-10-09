#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   scores.py
@Time    :   2024/09/18 10:43:17
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["PerformanceFn", "max_gap_target", "runtime_score"]

from collections.abc import Callable
import numpy as np

"""Performance Function type. From any sequence it calculates the performance score.
Returns:
    float: Performance score
"""
PerformanceFn = Callable[[np.ndarray], np.ndarray]


def max_gap_target(scores: np.ndarray) -> np.ndarray:
    """Maximum gap to target.
    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.
    Use this metric to generate instances that are EASY to solve by the target algorithm

    Args:
        scores (np.ndarray[float]): Scores of each solver over every instances. It is expected
        that the first value is the score of the target.

    Returns:
        np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"Expected a 2d numpy array (i, s). Where `i` is the number of instances, `s` the number of solvers in the portfolio. Instead, scores have shape: {scores.shape}"
        )
    return scores[:, 0] - np.max(scores[:, 1:], axis=1)


def runtime_score(scores: np.ndarray) -> np.ndarray:
    """Runtime based metric.
        It tries to maximise the gap between the runing time of the target solver
        and the other solvers in the portfolio. Use this metric with exact solvers
        which provide the same objective values for an instance.

    Args:
            scores (np.ndarray[float]): Scores of each solver over every instances. It is expected
            that the first value is the score of the target.

        Returns:
            np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    return np.min(scores[:, 1:], axis=1) - scores[:, 0]
