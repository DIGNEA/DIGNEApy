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


from typing import Callable

import numpy as np

from ..typing import PerformanceFn


def maximise_perf_gap_easy(
    scores: np.ndarray, aggregate_fn: Callable = np.mean
) -> np.ndarray:
    """Maximum performance bias gap to target.

    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.
    Use this metric to generate instances that are EASY to solve by the target algorithm

    Args:
        scores (np.ndarray[float]): Scores of each solver over every instances. It is expected
            that the first value is the score of the target.
        aggregate_fn (Callable): Function to aggregate the result of multiple repetitions.
            by default, it uses np.mean.

    Returns:
        np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if scores.ndim < 2 or scores.ndim > 3:
        raise ValueError(
            "Expected a 2d or 3d numpy array (i, solver, repetitions)"
            "Where `i` is the number of instances, `solver` the number of solvers in the portfolio, "
            "and repetitions is the number of repetitions performed by the solver on each instance."
            f" Instead, scores have shape: {scores.shape}"
        )
    elif scores.ndim == 2:
        return scores[:, 0] - np.max(scores[:, 1:], axis=1)
    else:
        aggregated = aggregate_fn(scores, axis=-1)
        return aggregated[:, 0] - np.max(aggregated[:, 1:], axis=-1)


def maximise_perf_gap_hard(
    scores: np.ndarray, aggregate_fn: Callable = np.mean
) -> np.ndarray:
    """Maximum performance bias gap between others and target.

    It tries to maximise the gap between the target solver
    and the other solvers in the portfolio.
    Use this metric to generate instances that are HARD to solve by the target algorithm

    Args:
        scores (np.ndarray[float]): Scores of each solver over every instances. It is expected
            that the first value is the score of the target.
        aggregate_fn (Callable): Function to aggregate the result of multiple repetitions.
            by default, it uses np.mean.

    Returns:
        np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if scores.ndim < 2 or scores.ndim > 3:
        raise ValueError(
            "Expected a 2d or 3d numpy array (i, solver, repetitions)"
            "Where `i` is the number of instances, `solver` the number of solvers in the portfolio, "
            "and repetitions is the number of repetitions performed by the solver on each instance."
            f" Instead, scores have shape: {scores.shape}"
        )
    if scores.ndim == 2:
        return max(scores[:, 0]) - np.min(scores[:, 1:], axis=1)
    else:
        aggregated = aggregate_fn(scores, axis=-1)
        return max(aggregated[:, 0]) - np.min(aggregated[:, 1:], axis=-1)


def maximise_runtime_gap(
    runtimes: np.ndarray, aggregate_fn: Callable = np.max
) -> np.ndarray:
    """Runtime based metric.

        It tries to maximise the gap between the running time of the target solver
        and the other solvers in the portfolio. Use this metric with exact solvers
        which provide the same objective values for an instance. The goal is that
        the instances generated would take less time to solve by the target solver.

    Args:
            runtimes (np.ndarray[float]): Running times of each solver over every instances.
                It is expected  that the first value is the score of the target.
            aggregate_fn (Callable): Function to aggregate the result of multiple repetitions.
                by default, it uses np.max.

        Returns:
            np.ndarray: Performance biases for every instance. Instance.p attribute.
    """
    if runtimes.ndim < 2 or runtimes.ndim > 3:
        raise ValueError(
            "Expected a 2d or 3d numpy array (i, solver, repetitions)"
            "Where `i` is the number of instances, `solver` the number of solvers in the portfolio, "
            "and repetitions is the number of repetitions performed by the solver on each instance."
            f" Instead, scores have shape: {runtimes.shape}"
        )
    if runtimes.ndim == 2:
        return np.min(runtimes[:, 1:], axis=1) - runtimes[:, 0]
    else:
        aggregated = aggregate_fn(runtimes, axis=-1)
        return np.min(aggregated[:, 1:], axis=-1) - aggregated[:, 0]
