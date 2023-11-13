#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   nn_novelty_search.py
@Time    :   2023/11/10 13:53:37
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""
from typing import List
import numpy as np
import pandas as pd
import keras
from keras import layers, activations, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deap import algorithms, base, benchmarks, cma, creator, tools


class TransformModel:
    def __init__(
        self,
        n_features: int,
        n_outputs: int = 2,
        input_activation=keras.activations.relu,
        output_activation=keras.activations.sigmoid,
        loss_function: str = "mse",
        optimizer: str = "sgd",
        metrics: List[str] = None,
    ):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.input_activation = input_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.hidden_nodes = np.ceil((self.n_features + self.n_outputs) / 2)
        self.n_weights = (
            (self.n_features * self.hidden_nodes)
            + (self.hidden_nodes * self.n_outputs)
            + self.hidden_nodes
            + self.n_outputs
        )

        self.model = models.Sequential(
            [
                layers.Dense(
                    self.n_features,
                    input_dim=self.n_features,
                    activation=self.input_activation,
                ),
                layers.Dense(
                    self.hidden_nodes,
                    activation=self.input_activation,
                ),
                layers.Dense(self.n_outputs, activation=self.output_activation),
            ]
        )

        self.model.compile(loss="mse", optimizer="sgd", metrics=["mse"])

    def train(self):
        pass

    def update_weights(self):
        pass

    def predict(self, individual=None, dataset=None):
        if not individual:
            msg = "The individual cannot be None in TransformModel predict"
            raise RuntimeError(msg)

        if not dataset:
            msg = "The dataset cannot be None in TransformModel predict"
            raise RuntimeError(msg)

        # set the weights according to the individual
        slice_0 = self.hidden_nodes * self.n_features
        slice_1 = slice_0 + (self.hidden_nodes * self.n_outputs)
        slice_2 = slice_1 + (self.hidden_nodes)

        first_layer_w = individual[:slice_0]
        second_layer_w = individual[slice_0:slice_1]
        first_layer_biases = np.array(individual[slice_1:slice_2])
        second_layer_biases = np.array(individual[slice_2:])

        # the weights layer needs to be in a 2d form  (featuresxhidden) for layer 1 and (hiddenx2) for layer 2
        w_1 = np.reshape(first_layer_w, (self.n_features, self.hidden_nodes))
        w_2 = np.reshape(second_layer_w, (self.hidden_nodes, self.n_outputs))

        # now we need to make an array of arrays  that contains the weights and the biases for each layer
        # as this is the form the get/set weights function use
        layer_1_weights = np.array([w_1, first_layer_biases], dtype="object")
        layer_2_weights = np.array([w_2, second_layer_biases], dtype="object")

        # and now lets reset the weights to the ones that came from the CMA-ES
        self.model.layers[0].set_weights(layer_1_weights)
        self.model.layers[1].set_weights(layer_2_weights)

        # x_train is always the 70% training set (which then gets further split)
        return self.model.predict(dataset)


class HyperCMA:
    __directions = ("minimise", "maximise")

    def __init__(
        self,
        dimension: int = 0,
        centroid: list = None,
        sigma: float = 5.0,
        lambda_: int = 0,
        generations: int = 250,
        direction: str = "maximise",
        seed: int = 42,
    ):
        self.dimension = dimension
        self.centroid = centroid if centroid is not None else [5.0] * self.dimension
        self.sigma = sigma
        self._lambda = lambda_ if lambda_ != 0 else 20 * self.dimension
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
        self.toolbox.register("evaluate", benchmarks.rastrigin)
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
        raise NotImplemented("")

    def __call__(self):
        algorithms.eaGenerateUpdate(
            self.toolbox, ngen=self.generations, stats=self.stats, halloffame=self.hof
        )
        return (self.hof[0], self.hof[0].fitness.values[0])
