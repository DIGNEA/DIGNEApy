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

# __all__ = [
#     "Transformer",
#     "SupportsTransform",
#     "nettrans",
#     "NNTuner",
# ]

__all__ = [
    "Transformer",
    "SupportsTransform",
]


# def __getattr__(attr):
#     if attr == "nettrans":
#         import digneapy.transformers._neural_networks as nettrans

#         return nettrans

#     if attr == "autoencoders":
#         import digneapy.transformers._autoencoders as autoencoders

#         return autoencoders

#     if attr == "tuner":
#         import digneapy.transformers._tuner as tuner

#         return tuner
