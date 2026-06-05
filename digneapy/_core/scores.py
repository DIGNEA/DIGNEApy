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


import numpy as np

from ..typing import PerformanceFn


def max_gap_target(portfolio_scores: np.ndarray) -> np.ndarray:
    """Maximum performance bias gap between target and others.

    Function to returns the fitness as the difference between the score of the target (position zero)
    and the maximum score of the other solvers in the portfolio.

    Use this metric to generate instances that are EASY to solve by the target algorithm. This is
    a metric to maximise.

    Args:
        portfolio_scores (np.ndarray[float]): Scores of each solver over every instances.
        It is expected that the first value is the score of the target.

    Returns:
        np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if portfolio_scores.ndim != 2:
        raise ValueError(
            "Expected a 2d numpy array (i, s). Where `i` is the number of instances, `s` the number of solvers in the portfolio. "
            f"Instead, scores have shape: {portfolio_scores.shape}."
        )
    return portfolio_scores[:, 0] - np.max(portfolio_scores[:, 1:], axis=1)


def runtime_score(portfolio_scores: np.ndarray) -> np.ndarray:
    """Minimum runtime gap between target and others.

    It tries to maximise the gap between the runtime of the target solver
    and the other solvers in the portfolio. Use this metric with *^exact** solvers
    which provide the same objective values for an instance.

    Args:
        portfolio_scores (np.ndarray[float]): Scores of each solver over every instances. It is expected
        that the first value is the score of the target.

    Returns:
        np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if portfolio_scores.ndim != 2:
        raise ValueError(
            "Expected a 2d numpy array (i, s). Where `i` is the number of instances, `s` the number of solvers in the portfolio. "
            f"Instead, scores have shape: {portfolio_scores.shape}."
        )
    return np.min(portfolio_scores[:, 1:], axis=1) - portfolio_scores[:, 0]
