#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_transformers.py
@Time    :   2023/11/16 08:58:15
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

import os
import pytest
import numpy as np
from digneapy.transformers import Transformer, NN


def test_NN_transformer_bpp():
    shapes = (11, 5, 2)
    activations = ("relu", "relu")
    expected_filename = "nn_transformer_bpp.keras"
    transformer = NN(expected_filename, shape=shapes, activations=activations)

    weights = np.random.random_sample(size=204)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    transformer.save()
    assert os.path.exists(os.path.join(os.path.curdir, expected_filename))
    os.remove(os.path.join(os.path.curdir, expected_filename))


def test_NN_transformer_kp():
    shapes = (8, 4, 2)
    activations = ("relu", "relu")
    transformer = NN("nn_transformer_kp.keras", shape=shapes, activations=activations)

    weights = np.random.random_sample(size=118)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)
