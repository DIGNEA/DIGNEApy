#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   perf_metrics.py
@Time    :   2024/06/07 14:24:19
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


from collections.abc import Sequence


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
