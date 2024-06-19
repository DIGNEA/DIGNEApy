#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _keras_nn.py
@Time    :   2024/06/07 13:57:23
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence
from typing import Optional

import keras
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler

from .base import Transformer


class KerasNN(Transformer):
    def __init__(
        self,
        name: str,
        input_shape: Sequence[int],
        shape: Sequence[int],
        activations: Sequence[Optional[str]],
        scale: bool = True,
    ):
        """Neural Network used to transform a space into another. This class uses a Tensorflow and Keras backend.

        Args:
            name (str): Name of the model to be saved with. Expected a .keras extension.
            shape (Tuple[int]): Tuple with the number of cells per layer.
            activations (Tuple[str]): Activation functions for each layer.
            scale (bool, optional): Includes scaler step before prediction. Defaults to True.

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
        self._scaler = StandardScaler() if scale else None

        self._model = keras.models.Sequential()
        self._model.add(keras.layers.InputLayer(shape=input_shape))
        for d, act in zip(shape, activations):
            self._model.add(keras.layers.Dense(d, activation=act))
        self._model.compile(
            loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.001)
        )

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
        expected = np.sum([np.prod(v.shape) for v in self._model.trainable_variables])
        if len(weights) != expected:
            msg = f"Error in the amount of weights in NN.update_weigths. Expected {expected} and got {len(weights)}"
            raise ValueError(msg)
        start = 0
        new_weights = []
        for v in self._model.trainable_variables:
            stop = start + np.prod(v.shape)
            new_weights.append(np.reshape(weights[start:stop], v.shape))
            start = stop

        reshaped_weights = np.array(new_weights, dtype=object)
        self._model.set_weights(reshaped_weights)
        return True

    def predict(self, X: npt.NDArray) -> np.ndarray:
        if len(X) == 0:
            msg = "X cannot be None in KerasNN predict"
            raise RuntimeError(msg)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        return self._model.predict(X, verbose=0)

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        return self.predict(X)
