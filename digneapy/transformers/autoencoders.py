#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   autoencoders.py
@Time    :   2024/05/28 10:06:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["KPEncoder"]

import os

os.environ["KERAS_BACKEND"] = "torch"

import joblib
import keras
import numpy as np
import numpy.typing as npt
from keras.utils import pad_sequences

from digneapy.transformers._base import Transformer


class KPEncoder(Transformer):
    _AVAILABLE_ENCODERS = (50, 100, 250, 500, 100, "var_2d", "var_8d", "var_best")
    _MAX_LENGTH = 2001
    _MODELS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/models/"

    def __init__(self, name: str = "KPEncoder", encoder: str | int = 50):
        super().__init__(name)

        if encoder not in KPEncoder._AVAILABLE_ENCODERS:
            raise ValueError(
                f"The encoding alternatives must be of type int and {KPEncoder._AVAILABLE_ENCODERS}"
            )

        self._encoder = encoder
        self._pad = (
            True if self._encoder in KPEncoder._AVAILABLE_ENCODERS[-3:] else False
        )
        pipeline_fn = (
            f"{KPEncoder._MODELS_PATH}pipeline_scaler_autoencoder_N_{self._encoder}.sav"
        )
        self._pipeline = joblib.load(pipeline_fn)

    def encode(self, X: npt.NDArray) -> np.ndarray:
        # Gets an array of instances
        # Scale and pad the instances
        # Encode them
        _X = X
        if self._pad:
            _X = pad_sequences(
                _X, padding="post", dtype="float32", maxlen=KPEncoder._MAX_LENGTH
            )
        return self._pipeline.predict(_X, verbose=0)

    def decode(self, X: npt.NDArray) -> np.ndarray:
        # # Gets an array of encoded instances
        # # Decode them
        # # Use scaler.inverse_transform() to get the instance back
        # X_decoded = self._decoder.predict(X, verbose=0)
        # X_decoded = self._scaler.inverse_transform(X_decoded)
        # return X_decoded
        raise NotImplementedError("Not implemented in this version")

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        return self.encode(X)
