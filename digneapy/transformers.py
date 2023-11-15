#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   transformers.py
@Time    :   2023/11/15 08:51:42
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

from typing import Any, List, Tuple, Callable
import numpy as np
import pandas as pd
import keras
from keras import layers, activations, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deap import algorithms, base, benchmarks, cma, creator, tools


class Transformer:
    def __init__(self, name: str):
        self._name = name

    def train(self, X: List[float]):
        raise NotImplemented("train method not implemented in Transformer")

    def predict(self, X: List[float]):
        raise NotImplemented("predict method not implemented in Transformer")

    def save(self):
        raise NotImplemented("save method not implemented in Transformer")


class NN(Transformer):
    def __init__(self, name: str, shape: Tuple[int], activations: Tuple[str]):
        super.__init__(name)
        if len(activations) != len(shape) - 1:
            msg = f"Expected {len(shape) - 1} activation functions but only got {len(activations)}"
            raise AttributeError(msg)
        self._shape = shape
        self._activations = activations

        model_layers = []
        for i, dim, act_fn in enumerate(zip(self._shape, self._activations)):
            if i == 0:
                layer = layers.Dense(
                    dim,
                    input_dim=dim,
                    activation=act_fn,
                )

            elif i == len(self._shape) - 1:
                layer = layers.Dense(
                    dim,
                )

            else:
                layer = layers.Dense(
                    dim,
                )

            model_layers.append(layer)

        self._model = models.Sequential(model_layers)
        lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1, decay_steps=10000, decay_rate=1e-4
        )
        self._model.compile(
            loss="mse", optimizer=keras.optimizers.SGD(learning_rate=lr_scheduler)
        )

    def save(self):
        self._model.save(self._name)

    def train(self):
        pass

    def update_weights(self, weights: List[float]):
        # TODO: Update to make generic --> Now only works for BPP
        w_0 = np.reshape(weights[:121], (11, 11))
        w_1 = np.reshape(weights[121:132], (11,))
        w_2 = np.reshape(weights[132:187], (11, 5))
        w_3 = np.reshape(weights[187:192], (5,))
        w_4 = np.reshape(weights[192:202], (5, 2))
        w_5 = np.reshape(weights[202:], (2,))
        reshaped_weights = np.array([w_0, w_1, w_2, w_3, w_4, w_5], dtype=object)
        self._model.set_weights(reshaped_weights)

    def predict(self, X: List):
        if not X:
            msg = "X cannot be None in NN predict"
            raise RuntimeError(msg)

        return self._model.predict(X, verbose=0)

    def __call__(self, X: List):
        return self.predict(X)


class HyperCMA:
    __directions = ("minimise", "maximise")

    def __init__(
        self,
        dimension: int = 0,
        centroid: list = None,
        sigma: float = 1.0,
        lambda_: int = 50,
        generations: int = 250,
        direction: str = "maximise",
        transformer: Transformer = None,
        experiment_work: Callable = None,
        seed: int = 42,
    ):
        if transformer is None or not isinstance(transformer, NN):
            raise AttributeError("transformer must be a NN object to run HyperCMA")
        if experiment_work is None:
            raise AttributeError(
                "experiment_work must be a callable object to run HyperCMA"
            )
        self.transformer = transformer
        self.experiment_work = experiment_work

        self.dimension = dimension
        self.centroid = centroid if centroid is not None else [0.0] * self.dimension
        self.sigma = sigma
        self._lambda = lambda_ if lambda_ != 0 else 50
        self.generations = generations
        self.seed = seed
        np.random.seed(self.seed)

        if direction not in self.__directions:
            msg = f"Direction: {direction} not available. Please choose between {self.__directions}"
            raise AttributeError(msg)

        self.direction = direction
        if self.direction == "maximise":
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        elif self.direction == "minimise":
            creator.create("Fitness", base.Fitness, weights=(-1.0,))

        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.evaluation)
        self.strategy = cma.Strategy(
            centroid=self.centroid, sigma=self.sigma, lambda_=self._lambda
        )
        self.toolbox.register("generate", self.strategy.generate, creator.Individual)
        self.toolbox.register("update", self.strategy.update)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluation(self, individual):
        """Evaluates a chromosome of weights for a NN
        to generate spaces in optimisation domains

        Args:
            individual (list): List of weights for a NN transformer

        Returns:
            float: Space coverage of the space create from the NN transformer
        """
        self.transformer.update_weights(individual)
        fitness = self.experiment_work(Transformer)
        return (fitness,)

    def __call__(self):
        algorithms.eaGenerateUpdate(
            self.toolbox, ngen=self.generations, stats=self.stats, halloffame=self.hof
        )
        return (self.hof[0], self.hof[0].fitness.values[0])
