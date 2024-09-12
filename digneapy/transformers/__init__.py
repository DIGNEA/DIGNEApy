#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/07 13:56:21
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

from digneapy.transformers._autoencoders import KPAE, KPAE50
from digneapy.transformers._base import SupportsTransform, Transformer
from digneapy.transformers._keras_nn import KerasNN
from digneapy.transformers._torch_nn import TorchNN
from digneapy.transformers._tuner import NNTuner

__all__ = [
    "Transformer",
    "SupportsTransform",
    "KerasNN",
    "TorchNN",
    "NNTuner",
    "KPAE",
    "KPAE50",
]
