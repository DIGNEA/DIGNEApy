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
import numpy as np
import numpy.typing as npt
from keras.utils import pad_sequences

from digneapy.transformers._base import Transformer


class KPEncoder(Transformer):
    _AVAILABLE_ENCODERS = ("50", "100", "500", "1000", "2000", "5000", "variable")
    _MAX_LENGTH = 2001
    _MODELS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/models/"

    def __init__(self, name: str = "KPEncoder", encoder: str = "variable"):
        super().__init__(name)

        if encoder not in KPEncoder._AVAILABLE_ENCODERS:
            raise ValueError(
                f"The encoding alternatives must be of: {KPEncoder._AVAILABLE_ENCODERS}"
            )

        self._encoder = encoder
        self._pad = True if self._encoder == "variable" else False
        pipeline_fn = (
            f"{KPEncoder._MODELS_PATH}pipeline_scaler_AE_N_{self._encoder}.sav"
        )
        self._pipeline = joblib.load(pipeline_fn)

    @property
    def encoding(self) -> int:
        return 2

    def encode(self, X: npt.NDArray) -> np.ndarray:
        # Gets an array of instances
        # Scale and pad the instances
        # Encode them
        _X = np.asarray(X)
        if self._pad:
            _X = pad_sequences(
                _X, padding="post", dtype="float32", maxlen=KPEncoder._MAX_LENGTH
            )
        return self._pipeline.predict(_X, verbose=0)

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        return self.encode(X)
