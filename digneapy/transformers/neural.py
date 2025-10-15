#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _neural_networks.py
@Time    :   2024/09/12 14:21:29
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

__all__ = ["NNEncoder"]

from collections.abc import Sequence
from typing import Callable, Optional

import keras
import numpy as np
import numpy.typing as npt

from ._base import Transformer


class NNEncoder(Transformer):
    def __init__(
        self,
        name: str,
        input_shape: Sequence[int],
        shape: Sequence[int],
        activations: Sequence[Optional[str]],
        evaluation_metric: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[keras.Optimizer] = keras.optimizers.Nadam(),
    ):
        """Neural Network used to transform a space into another. This class uses a Keras backend.

        Args:
            name (str): Name of the model to be saved with. Expected a .keras extension.
            shape (Tuple[int]): Tuple with the number of cells per layer.
            activations (Tuple[str]): Activation functions for each layer.
        Raises:
            ValueError: Raises if any attribute is not valid.
        """
        if len(activations) != len(shape):
            msg = f"Expected {len(shape)} activation functions but only got {len(activations)}"
            raise ValueError(msg)
        if not name.endswith(".keras"):
            name = name + ".keras"

        super().__init__(name)

        self.input_shape = input_shape
        self._shape = shape
        self._activations = activations
        self._eval_metric = evaluation_metric
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        self._model = keras.models.Sequential()
        self._model.add(keras.layers.InputLayer(shape=input_shape))
        for d, act in zip(shape, activations):
            self._model.add(keras.layers.Dense(d, activation=act))
        self._model.compile(
            loss=self._loss_fn, optimizer=optimizer, metrics=[self._eval_metric]
        )
        self._expected_shapes = [v.shape for v in self._model.trainable_variables]
        self._expected_sizes = [np.prod(s) for s in self._expected_shapes]
        self._size = np.sum(self._expected_sizes)

    def __str__(self) -> str:
        tokens = []
        self._model.summary(print_fn=lambda x: tokens.append(x))
        return "\n".join(tokens)

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, filename: Optional[str] = None):
        if filename is not None:
            self._model.save(filename)
        else:
            self._model.save(self._name)

    def update_weights(self, weights: Sequence[float]):
        if len(weights) != self._size:
            msg = f"Error in the amount of weights in NN.update_weigths. Expected {self._size} and got {len(weights)}"
            raise ValueError(msg)

        new_weights = [None] * len(self._expected_shapes)
        idx = 0
        i = 0
        for shape, size in zip(self._expected_shapes, self._expected_sizes):
            new_weights[i] = np.reshape(weights[idx : idx + size], shape)
            idx += size
            i += 1

        self._model.set_weights(new_weights)
        return True

    def predict(self, x: npt.NDArray, batch_size: int = 1024) -> np.ndarray:
        if x is None or len(x) == 0:
            msg = "x cannot be None in KerasNN predict"
            raise RuntimeError(msg)
        if isinstance(x, list):
            x = np.vstack(x)
        elif x.ndim == 1:
            x.reshape(1, -1)

        return self._model.predict(x, batch_size=batch_size, verbose=2)

    def __call__(self, x: npt.NDArray) -> np.ndarray:
        return self.predict(x)
