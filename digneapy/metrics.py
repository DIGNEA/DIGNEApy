#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   metrics.py
@Time    :   2024/09/17 14:51:11
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["PerformanceFn", "default_performance_metric", "pisinger_performance_metric"]

from collections.abc import Callable, Sequence

"""Performance Function type. From any sequence it calculates the performance score.
Returns:
    float: Performance score
"""
PerformanceFn = Callable[[Sequence], float]


def default_performance_metric(scores: Sequence[float]) -> float:
    """Default performace metric for the instances.
    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.

    Args:
        scores (Iterable[float]): Scores of each solver over an instance. It is expected
        that the first value is the score of the target.

    Returns:
        float: Performance value for an instance. Instance.p attribute.
    """
    return scores[0] - max(scores[1:])


def pisinger_performance_metric(scores: Sequence[float]) -> float:
    """Pisinger Solvers performace metric for the instances.
    It tries to maximise the gap between the runing time of the target solver
    and the other solvers in the portfolio.

    Args:
        scores (Iterable[float]): Running time of each solver over an instance. It is expected
        that the first value is the score of the target.

    Returns:
        float: Performance value for an instance. Instance.p attribute.
    """
    return min(scores[1:]) - scores[0]