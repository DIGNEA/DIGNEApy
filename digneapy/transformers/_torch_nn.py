#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   torch_nn.py
@Time    :   2024/06/07 13:58:32
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._base_transformer import Transformer
from collections.abc import Sequence
from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


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

    def __str__(self) -> str:
        return self._model.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, filename: Optional[str] = None):
        name = filename if filename is not None else self._name
        torch.save(self._model.state_dict(), name)

    def train(self):
        pass

    def update_weights(self, parameters: Sequence[float]):
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

    def predict(self, X: Sequence):
        return self.forward(X)

    def forward(self, X):
        """This is a necessary method for the PyTorch module.
        It works as a predict method in Keras

        Args:
            X (Sequence): Sequence of instances to evaluate and predict their descriptor

        Raises:
            RuntimeError: If Sequence is empty

        Returns:
            Numpy.ndarray: Descriptor of the instances
        """
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
