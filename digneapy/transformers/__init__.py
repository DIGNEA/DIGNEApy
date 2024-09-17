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

from digneapy.transformers._base import SupportsTransform, Transformer

# from ._nnets import KerasNN, TorchNN
# from ._tuner import NNTuner

__all__ = ["Transformer", "SupportsTransform"]


def __getattr__(attr_name):
    if attr_name == "neural":
        import digneapy.transformers.neural as neural

        return neural

    elif attr_name == "autoencoders":
        import digneapy.transformers.autoencoders as autoencoders

        return autoencoders

    elif attr_name == "tuner":
        import digneapy.transformers.tuner as tuner

        return tuner
    else:
        raise ImportError(
            f"module 'digneapy.solvers.transformers' has no attribute {attr_name}"
        )
