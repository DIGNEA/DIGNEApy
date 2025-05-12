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

__all__ = ["NNTuner", "Tunner"]

from collections.abc import Callable, Sequence
from multiprocessing.pool import ThreadPool as Pool
from typing import Optional, Tuple

import fcmaes.cmaes
import fcmaes.crfmnes
import fcmaes.crfmnescpp
import numpy as np
from deap import algorithms, base, cma, creator, tools

from digneapy._core.types import RNG

from .._core._constants import Direction
from .neural import KerasNN, TorchNN
import fcmaes

from fcmaes.optimizer import wrapper
from scipy.optimize import Bounds

class NNTuner:
    def __init__(
        self,
        eval_fn: Callable,
        dimension: int,
        transformer: KerasNN | TorchNN,
        centroid: Optional[Sequence[float]] = None,
        sigma: float = 1.0,
        lambda_: int = 50,
        generations: int = 250,
        direction: Direction = Direction.MAXIMISE,
        n_jobs: int = 1,
    ):

        if eval_fn is None:
            raise ValueError(
                "eval_fn cannot be None in NNTuner. Please give a valid evaluation function."
            )
        self.eval_fn = eval_fn

        self.dimension = dimension
        if transformer is None:
            raise ValueError(
                "transformer cannot be None in NNTuner. Please give a valid transformer."
            )
        self.transformer = transformer
        self.centroid = centroid if centroid is not None else [0.0] * self.dimension
        self.sigma = sigma
        self._lambda = lambda_ if lambda_ != 0 else 50
        self.generations = generations
        self.__performed_gens = 0  # These vars are used to save the data in CSV files
        self.__evaluated_inds = 0

        if not isinstance(direction, Direction):
            msg = f"Direction: {direction} not available. Please choose between {Direction.values()}"
            raise ValueError(msg)

        self.direction = direction
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.evaluation)
        self.strategy = cma.Strategy(
            centroid=self.centroid, sigma=self.sigma, lambda_=self._lambda
        )
        if self.direction == Direction.MAXIMISE:
            self.toolbox.register("generate", self.strategy.generate, creator.IndMax)
        else:
            self.toolbox.register("generate", self.strategy.generate, creator.IndMin)
        self.toolbox.register("update", self.strategy.update)
        if n_jobs < 1:
            msg = "The number of jobs must be at least 1."
            raise ValueError(msg)
        elif n_jobs > 1:
            self.n_processors = n_jobs
            self.pool = Pool(processes=self.n_processors)
            self.toolbox.register("map", self.pool.map)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluation(self, individual: Sequence[float]) -> tuple[float]:
        """Evaluates a chromosome of weights for a NN
        to generate spaces in optimisation domains

        Args:
            individual (Sequence[float]): Sequence of weights for a NN transformer

        Returns:
            tuple[float]: Space coverage of the space create from the NN transformer
        """
        self.transformer.update_weights(individual)
        self.__evaluated_inds += 1
        if self.__evaluated_inds == self._lambda:
            self.__performed_gens += 1
            self.__evaluated_inds = 0

        fitness = self.eval_fn(self.transformer)
        return (fitness,)

    def __call__(self):
        population, logbook = algorithms.eaGenerateUpdate(
            self.toolbox,
            ngen=self.generations,
            stats=self.stats,
            halloffame=self.hof,
            verbose=True,
        )
        return (self.hof[0], population, logbook)


class Tunner(RNG):
    def __init__(
        self,
        dimension: int,
        ranges: Tuple[float, float],
        lambda_: int = 100,
        generations: int = 10,
        seed: int = 42,
    ):
        self._dimension = dimension
        self._bounds = Bounds([ranges[0]] * self._dimension, [ranges[1]] * self._dimension)
        self._pop_size = lambda_
        self._max_generations = generations
        self._seed = seed
        self.initialize_rng(seed=seed)
    def __call__(self, eval_fn: Callable):
        print(f'Starting the tunning process with {self._pop_size} individuals and {self._max_generations} generations.')
        solutions = fcmaes.cmaes.minimize(
                wrapper(eval_fn),
                x0=self._rng.uniform(self._bounds.lb, self._bounds.ub, size=self._dimension),
                max_iterations=(self._max_generations),
                #bounds=self._bounds,
                rg=self._rng, workers=4
        )      
        return solutions