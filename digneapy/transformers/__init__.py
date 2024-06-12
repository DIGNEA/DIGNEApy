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

from digneapy.transformers.base import Transformer
from digneapy.transformers.keras_nn import KerasNN
from digneapy.transformers.autoencoders import KPAE
from digneapy.transformers.tuner import NNTuner
from digneapy.transformers.torch_nn import TorchNN

# __all__ = ["Transformer", "KerasNN", "TorchNN", "NNTuner", "KPAE"]
