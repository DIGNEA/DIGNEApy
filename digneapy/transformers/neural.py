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

__all__ = ["KerasNN", "TorchNN"]

import os

os.environ["KERAS_BACKEND"] = "torch"

from collections.abc import Sequence
from typing import Callable, Optional

import keras
import numpy as np
import numpy.typing as npt
import torch

from ._base import Transformer


class KerasNN(Transformer):
    def __init__(
        self,
        name: str,
        input_shape: Sequence[int],
        shape: Sequence[int],
        activations: Sequence[Optional[str]],
        evaluation_metric: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[keras.Optimizer] = keras.optimizers.Adam(),
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
            new_weights[i] = np.reshape(weights[idx:idx+size], shape)
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
    
    


class TorchNN(Transformer, torch.nn.Module):
    def __init__(
        self,
        name: str,
        input_size: int,
        shape: Sequence[int],
        output_size: int,
    ):
        """Neural Network used to transform a space into another. This class uses a PyTorch backend.

        Args:
            name (str): Name of the model to be saved with. Expected a .torch extension.
            input_size (int): Number of neurons in the input layer.
            shape (Tuple[int]): Tuple with the number of cells per layer.
            output_size (int): Number of neurons in the output layer.
        Raises:
            ValueError: Raises if any attribute is not valid.
        """

        if not name.endswith(".torch"):
            name = name + ".torch"

        Transformer.__init__(self, name)
        torch.nn.Module.__init__(self)
        self._model = torch.nn.ModuleList([torch.nn.Linear(input_size, shape[0])])
        self._model.append(torch.nn.ReLU())
        for i in range(len(shape[1:-1])):
            self._model.append(torch.nn.Linear(shape[i], shape[i + 1]))
            self._model.append(torch.nn.ReLU())

        self._model.append(torch.nn.Linear(shape[-1], output_size))

    def __str__(self) -> str:
        return self._model.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, filename: Optional[str] = None):
        name = filename if filename is not None else self._name
        torch.save(self._model.state_dict(), name)

    def update_weights(self, parameters: Sequence[float]):
        expected = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if len(parameters) != expected:
            msg = f"Error in the amount of weights in NN.update_weigths. Expected {expected} and got {len(parameters)}"
            raise ValueError(msg)
        start = 0
        for layer in self._model.children():
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight
                stop = start + np.sum(
                    list(len(weights[i]) for i in range(len(weights)))
                )

                w_ = torch.Tensor(
                    np.array(parameters[start:stop]).reshape(
                        len(weights), len(weights[0])
                    )
                )
                start = stop
                stop = start + layer.out_features
                biases_ = torch.Tensor(np.array(parameters[start:stop]))

                layer.weight.data = w_
                layer.bias.data = biases_
                start = stop

        return True

    def predict(self, x: npt.NDArray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: npt.NDArray) -> np.ndarray:
        """This is a necessary method for the PyTorch module.
        It works as a predict method in Keras

        Args:
            x (npt.NDArray): Sequence of instances to evaluate and predict their descriptor

        Raises:
            RuntimeError: If Sequence is empty

        Returns:
            Numpy.ndarray: Descriptor of the instances
        """
        if len(x) == 0:
            msg = "x cannot be None in TorchNN forward"
            raise RuntimeError(msg)
        x = torch.tensor(x, dtype=torch.float32)
        y = x
        for layer in self._model:
            y = layer(y)
        y = y.detach().numpy()
        return y

    def __call__(self, x: npt.NDArray) -> np.ndarray:
        return self.forward(x)
