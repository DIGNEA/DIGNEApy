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

from collections.abc import Sequence
from multiprocessing.pool import ThreadPool as Pool
from typing import Callable, Optional

import numpy as np
from deap import algorithms, base, cma, creator, tools

from digneapy import Direction

from .base import Transformer
from .keras_nn import KerasNN
from .torch_nn import TorchNN


class NNTuner:
    def __init__(
        self,
        transformer: KerasNN | TorchNN,
        eval_fn: Callable,
        dimension: int,
        centroid: Optional[Sequence[float]] = None,
        sigma: float = 1.0,
        lambda_: int = 50,
        generations: int = 250,
        direction: Direction = Direction.MAXIMISE,
        n_jobs: int = 1,
    ):
        if transformer is None or not issubclass(transformer.__class__, Transformer):
            raise AttributeError(
                "transformer must be a subclass of KerasNN or TorchNN object to run MetaEA"
            )
        if eval_fn is None:
            raise AttributeError(
                "experiment_work must be a callable object to run MetaEA"
            )
        self.transformer = transformer
        self.eval_fn = eval_fn

        self.dimension = dimension
        self.centroid = centroid if centroid is not None else [0.0] * self.dimension
        self.sigma = sigma
        self._lambda = lambda_ if lambda_ != 0 else 50
        self.generations = generations
        self.__performed_gens = 0  # These vars are used to save the data in CSV files
        self.__evaluated_inds = 0

        if not isinstance(direction, Direction):
            msg = f"Direction: {direction} not available. Please choose between {Direction.values()}"
            raise AttributeError(msg)

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
            raise AttributeError(msg)
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
        filename = f"dataset_generation_{self.__performed_gens}_individual_{self.__evaluated_inds}.csv"
        self.__evaluated_inds += 1
        if self.__evaluated_inds == self._lambda:
            self.__performed_gens += 1
            self.__evaluated_inds = 0

        fitness = self.eval_fn(self.transformer, filename)
        return (fitness,)

    def __call__(self):
        population, logbook = algorithms.eaGenerateUpdate(
            self.toolbox,
            ngen=self.generations,
            stats=self.stats,
            halloffame=self.hof,
        )
        return (self.hof[0], population, logbook)