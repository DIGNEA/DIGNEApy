#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   pca.py
@Time    :   2025/04/07 09:07:49
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import pickle
import warnings

import numpy as np
import numpy.typing as npt
from pathlib import Path
from digneapy.transformers._base import Transformer

warnings.filterwarnings("ignore")


class PCAEncoder(Transformer):
    _PATH = Path(__file__) / "models/"

    def __init__(self, name: str = "PCAEncoder"):
        super().__init__(name)
        pipeline_fn = f"{PCAEncoder._PATH}pipeline_8D_2D_balanced_for_kp.pkl"
        self._pipeline = pickle.load(open(pipeline_fn, "rb"))

    @property
    def encoding(self) -> int:
        return 2

    def __call__(self, X: npt.NDArray) -> np.ndarray:
        return self._pipeline.transform(np.vstack(X))
