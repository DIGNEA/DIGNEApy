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

import os
import pickle

import keras
import numpy as np
import numpy.typing as npt
from keras.utils import pad_sequences

from digneapy.transformers.base import Transformer


class KPAE(Transformer):
    _ENCODINGS = (2, 8)
    _MAX_LENGTH = 2001
    _2D_AE = "2D_encoding/best_kp_ae_bayesian_latent_dim_2D_lr_one_cycle_training.keras"
    _2D_ENC = "2D_encoding/best_kp_ae_bayesian_latent_dim_2D_lr_one_cycle_training_encoder.keras"
    _2D_DEC = "2D_encoding/best_kp_ae_bayesian_latent_dim_2D_lr_one_cycle_training_decoder.keras"

    _8D_AE = "8D_encoding/best_kp_ae_bayesian_8D_lr_one_cycle_training.keras"
    _8D_ENC = "8D_encoding/best_kp_ae_bayesian_8D_lr_one_cycle_training_encoder.keras"
    _8D_DEC = "8D_encoding/best_kp_ae_bayesian_8D_lr_one_cycle_training_decoder.keras"

    _MODELS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/models/"

    def __init__(self, name: str = "KP_AE", encoding: int = 2, padding: bool = False):
        super().__init__(name)

        if encoding not in KPAE._ENCODINGS:
            raise ValueError(
                f"The encoding alternatives must be of type int and {KPAE._ENCODINGS}"
            )

        self._encoding = encoding
        self._pad = padding
        if self._encoding == 2:
            self.ae_path = KPAE._2D_AE
            self.enc_path = KPAE._2D_ENC
            self.dec_path = KPAE._2D_DEC
        else:
            self.ae_path = KPAE._8D_AE
            self.enc_path = KPAE._8D_ENC
            self.dec_path = KPAE._8D_DEC

        with open(f"{KPAE._MODELS_PATH}kp_scaler_for_ae_different_N.pkl", "rb") as f:
            self._scaler = pickle.load(f)

        self._model = keras.models.load_model(f"{KPAE._MODELS_PATH}/{self.ae_path}")
        self._encoder = keras.models.load_model(f"{KPAE._MODELS_PATH}/{self.enc_path}")
        self._decoder = keras.models.load_model(f"{KPAE._MODELS_PATH}/{self.dec_path}")

    def _preprocess(self, X: npt.NDArray) -> np.ndarray:
        # Scale and pad the instances before using AE
        # We pad the data to maximum allowed length
        _X = X
        if self._pad:
            _X = pad_sequences(
                _X, padding="post", dtype="float32", maxlen=KPAE._MAX_LENGTH
            )
        _X = self._scaler.transform(_X)
        return _X

    def encode(self, X: npt.NDArray) -> np.ndarray:
        # Gets an array of instances
        # Scale and pad the instances
        # Encode them
        _X = self._preprocess(X)
        return self._encoder.predict(_X, verbose=0)

    def decode(self, X: npt.NDArray) -> np.ndarray:
        # Gets an array of encoded instances
        # Decode them
        # Use scaler.inverse_transform() to get the instance back
        X_decoded = self._decoder.predict(X, verbose=0)
        X_decoded = self._scaler.inverse_transform(X_decoded)
        return X_decoded

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        return self.encode(X)

class KPAE50(KPAE):
    def __init__(self):
        super().__init__("KPAE50", 2, False)
        self._model = keras.models.load_model(f"{KPAE._MODELS_PATH}N_50_2D_encoding/best_kp_ae_N_50_2D_lr_one_cycle_training.keras")
        self._encoder = keras.models.load_model(f"{KPAE._MODELS_PATH}N_50_2D_encoding/best_kp_ae_N_50_2D_lr_one_cycle_training_encoder.keras")
        self._decoder = keras.models.load_model(f"{KPAE._MODELS_PATH}N_50_2D_encoding/best_kp_ae_N_50_2D_lr_one_cycle_training_decoder.keras")
        with open(f"{KPAE._MODELS_PATH}kp_scaler_N_50.pkl", "rb") as f:
            self._scaler = pickle.load(f)