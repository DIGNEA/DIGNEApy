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

import numpy as np
from fcmaes import crfmnes
from fcmaes.optimizer import wrapper
from scipy.optimize import Bounds

type TunerFn = Callable[[np.ndarray], np.float64]


class Tuner:
    def __init__(
        self,
        dimension: int,
        ranges: Tuple[float, float],
        lambda_: int = 100,
        evaluations: int = 10,
        seed: Optional[int | np.random.SeedSequence] = None,
        workers: int = 1,
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

    def __call__(self, eval_fn: TunerFn):
        if eval_fn is None:
            raise ValueError("eval_fn cannot be None in Tuner.__call__")
        print(
            f"""Starting the tunning process with:
                    - Pop size: {self._pop_size} individuals 
                    - Evaluations: {self._max_evals}
                    - Workers: {self.workers}\n"""
        )
        solutions = crfmnes.minimize(
            wrapper(eval_fn),
            x0=self._rng.uniform(
                self._bounds.lb, self._bounds.ub, size=self._dimension
            ),
            max_evaluations=self._max_evals,
            popsize=self._pop_size,
            bounds=self._bounds,
            rg=self._rng,
            workers=self.workers,
        )

        return solutions
