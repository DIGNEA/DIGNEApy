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

from typing import List, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras


class Transformer:
    def __init__(self, name: str):
        self._name = name

    def train(self, X: List[float]):
        raise NotImplemented("train method not implemented in Transformer")

    def predict(self, X: List[float]):
        raise NotImplemented("predict method not implemented in Transformer")

    def __call__(self, X: List[float]):
        raise NotImplemented("__call__ method not implemented in Transformer")

    def save(self):
        raise NotImplemented("save method not implemented in Transformer")


class TorchNN(Transformer, torch.nn.Module):

    def __init__(
        self,
        name: str,
        input_size: int,
        shape: Tuple[int],
        output_size: int,
        scale: bool = True,
    ):
        """Neural Network used to transform a space into another. This class uses a PyTorch backend.

        Args:
            name (str): Name of the model to be saved with. Expected a .torch extension.
            input_size (int): Number of neurons in the input layer.
            shape (Tuple[int]): Tuple with the number of cells per layer.
            output_size (int): Number of neurons in the output layer.
            scale (bool, optional): Includes scaler step before prediction. Defaults to True.
        Raises:
            AttributeError: Raises if any attribute is not valid.
        """

        if not name.endswith(".torch"):
            name = name + ".torch"

        Transformer.__init__(self, name)
        torch.nn.Module.__init__(self)
        self._scaler = StandardScaler() if scale else None
        self._model = torch.nn.ModuleList([torch.nn.Linear(input_size, shape[0])])
        self._model.append(torch.nn.ReLU())
        for i in range(len(shape[1:-1])):
            self._model.append(torch.nn.Linear(shape[i], shape[i + 1]))
            self._model.append(torch.nn.ReLU())

        self._model.append(torch.nn.Linear(shape[-1], output_size))

    def __str__(self):
        return self._model.__str__()

    def __repr__(self):
        return self.__str__()

    def save(self, filename: str = None):
        name = filename if filename is not None else self._name
        torch.save(self._model.state_dict(), name)

    def train(self):
        pass

    def update_weights(self, parameters: List[float]):
        expected = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if len(parameters) != expected:
            msg = f"Error in the amount of weights in NN.update_weigths. Expected {expected} and got {len(parameters)}"
            raise AttributeError(msg)
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

    def predict(self, X: List):
        return self.forward(X)

    def forward(self, X):
        if len(X) == 0:
            msg = "X cannot be None in TorchNN forward"
            raise RuntimeError(msg)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = X
        for layer in self._model:
            y = layer(y)
        y = y.detach().numpy()
        return y

    def __call__(self, X):
        return self.forward(X)


class KerasNN(Transformer):
    def __init__(
        self,
        name: str,
        input_shape: Tuple[int],
        shape: Tuple[int],
        activations: Tuple[str],
        scale: bool = True,
    ):
        """Neural Network used to transform a space into another. This class uses a Tensorflow and Keras backend.

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
        self.input_shape = input_shape
        self._shape = shape
        self._activations = activations
        self._scaler = StandardScaler() if scale else None

        self._model = self.__build_model(input_shape, shape, activations)

    def __build_model(self, input_shape, shape, activations):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for d, act in zip(shape, activations):
            model.add(keras.layers.Dense(d, activation=act))
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.001))
        return model

    def __str__(self):
        tokens = []
        self._model.summary(print_fn=lambda x: tokens.append(x))
        return "\n".join(tokens)

    def __repr__(self):
        return self.__str__()

    def save(self, filename: str = None):
        if filename is not None:
            self._model.save(filename)
        else:
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
            msg = "X cannot be None in KerasNN predict"
            raise RuntimeError(msg)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        return self._model.predict(X, verbose=0)

    def __call__(self, X: List):
        return self.predict(X)
