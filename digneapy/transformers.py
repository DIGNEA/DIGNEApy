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
from keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deap import algorithms, base, cma, creator, tools
import keras.backend as K


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
    def __init__(
        self,
        name: str,
        shape: Tuple[int],
        activations: Tuple[str],
        scale: bool = True,
    ):
        """Neural Network used to transform a space into another.

        Args:
            name (str): Name of the model to be saved with. Expected a .keras extension.
            shape (Tuple[int]): Tuple with the number of cells per layer.
            activations (Tuple[str]): Activation functions for each layer.
            scale (bool, optional): Includes scaler step before prediction. Defaults to True.

        Raises:
            AttributeError: Raises if any attribute is not valid.
        """
        if len(activations) != len(shape):
            msg = f"Expected {len(shape)} activation functions but only got {len(activations)}"
            raise AttributeError(msg)
        if not name.endswith(".keras"):
            name = name + ".keras"

        super().__init__(name)
        self._shape = shape
        self._activations = activations
        self._scaler = StandardScaler() if scale else None
        model_layers = [
            layers.Dense(dim, input_dim=dim, activation=act_fn)
            for (dim, act_fn) in zip(self._shape, self._activations)
        ]

        assert len(model_layers) == len(shape)
        self._model = models.Sequential(model_layers)
        self._model.compile(
            loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.001)
        )

    def __str__(self):
        tokens = []
        self._model.summary(print_fn=lambda x: tokens.append(x))
        return "\n".join(tokens)

    def __repr__(self):
        return self.__str__()

    def save(self):
        self._model.save(self._name)

    def train(self):
        pass

    def update_weights(self, weights: List[float]):
        expected = np.sum([np.prod(v.shape) for v in self._model.trainable_variables])
        if len(weights) != expected:
            msg = f"Error in the amount of weights in NN.update_weigths. Expected {expected} and got {len(weights)}"
            raise AttributeError(msg)
        start = 0
        new_weights = []
        for v in self._model.trainable_variables:
            stop = start + np.prod(v.shape)
            new_weights.append(np.reshape(weights[start:stop], v.shape))
            start = stop

        reshaped_weights = np.array(new_weights, dtype=object)
        self._model.set_weights(reshaped_weights)
        return True

    def predict(self, X: List):
        if len(X) == 0:
            msg = "X cannot be None in NN predict"
            raise RuntimeError(msg)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
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
        population, logbook = algorithms.eaGenerateUpdate(
            self.toolbox, ngen=self.generations, stats=self.stats, halloffame=self.hof
        )
        return (self.hof[0], population, logbook)