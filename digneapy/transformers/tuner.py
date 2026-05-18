#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   meta_ea.py
@Time    :   2024/04/25 09:54:42
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["Tuner", "TunerFn"]

from collections.abc import Callable
from typing import Optional, Tuple

import cma
import numpy as np
from cma.evolution_strategy import CMAEvolutionStrategyResult2
from scipy.optimize import Bounds

type TunerFn = Callable[[np.ndarray], np.float64]


class Tuner:
    def __init__(
        self,
        dimension: int,
        ranges: Tuple[float, float],
        lambda_: int = 100,
        sigma: float = 0.5,
        evaluations: int = 10,
        workers: int = 1,
        seed: Optional[int | np.random.SeedSequence] = None,
    ):
        if any(param < 0 for param in (dimension, lambda_, evaluations, workers)):
            raise ValueError(
                f"These parameters cannot be negative:\n"
                f"\t- dimension. Got {dimension}"
                f"\t- lambda_ (Pop size). Got {lambda_}"
                f"\t- evaluations. Got {evaluations}"
                f"\t- workers. Got {workers}"
            )
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._dimension = dimension
        self._bounds = Bounds(
            [ranges[0]] * self._dimension, [ranges[1]] * self._dimension
        )
        self._pop_size = lambda_
        self._max_evals = evaluations
        self.workers = workers
        self._x0 = self._rng.uniform(
            self._bounds.lb, self._bounds.ub, size=self._dimension
        )
        self._sigma = sigma
        self._strategy = None

    def __call__(self, eval_fn: TunerFn) -> CMAEvolutionStrategyResult2:
        if eval_fn is None:
            raise ValueError("eval_fn cannot be None in Tuner.__call__")
        print(
            f"""Starting the tunning process with:
                    - Pop size: {self._pop_size} individuals 
                    - Evaluations: {self._max_evals}
                    - Workers: {self.workers}\n"""
        )
        self._strategy = cma.CMAEvolutionStrategy(
            self._x0,
            self._sigma,
            inopts={"popsize": self._pop_size, "maxfevals": self._max_evals},
        )
        self._strategy = self._strategy.optimize(
            eval_fn, self._max_evals, n_jobs=self.workers
        )
        solution = self._strategy.result
        return solution
