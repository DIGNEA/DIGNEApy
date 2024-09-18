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

from collections.abc import Callable, Sequence

"""Performance Function type. From any sequence it calculates the performance score.
Returns:
    float: Performance score
"""
PerformanceFn = Callable[[Sequence], float]


def max_gap_target(scores: Sequence[float]) -> float:
    """Maximum gap to target.
    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.
    Use this metric to generate instances that are EASY to solve by the target algorithm

    Args:
        scores (Iterable[float]): Scores of each solver over an instance. It is expected
        that the first value is the score of the target.

    Returns:
        float: Performance value for an instance. Instance.p attribute.
    """
    return scores[0] - max(scores[1:])


def runtime_score(scores: Sequence[float]) -> float:
    """Runtime based metric.
    It tries to maximise the gap between the runing time of the target solver
    and the other solvers in the portfolio. Use this metric with exact solvers
    which provide the same objective values for an instance.

    Args:
        scores (Iterable[float]): Running time of each solver over an instance. It is expected
        that the first value is the score of the target.

    Returns:
        float: Performance value for an instance. Instance.p attribute.
    """
    return min(scores[1:]) - scores[0]
