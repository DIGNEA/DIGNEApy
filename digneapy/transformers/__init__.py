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

from .pca import PCAEncoder
from .protocol import Transformer

__transformer_submodules = {"neural", "autoencoders", "tuner"}

__all__ = ["Transformer", "PCAEncoder"]

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Lazy import function
def __getattr__(attr_name):
    import importlib
    import sys

    if _TORCH_AVAILABLE:
        if attr_name in __transformer_submodules:
            full_name = f"digneapy.transformers.{attr_name}"
            submodule = importlib.import_module(full_name)
            sys.modules[full_name] = submodule
            return submodule

        else:
            raise ImportError(
                f"module digneapy.transformers has no attribute {attr_name}"
            )
    else:
        raise NotImplementedError(
            "This module is not available without a correct PyTorch installation"
        )
