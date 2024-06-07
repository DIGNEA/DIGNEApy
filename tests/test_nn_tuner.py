#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_meta_ea.py
@Time    :   2024/04/25 09:59:15
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""
from __future__ import unicode_literals

import os
import pytest
import numpy as np
import pandas as pd
from digneapy.transformers import NNTuner, KerasNN
from sklearn.metrics import mean_squared_error

dir, _ = os.path.split(__file__)

filename = "eig_bpp_instances_only_features.csv"
features = [
    "capacity",
    "huge",
    "large",
    "max",
    "mean",
    "median",
    "medium",
    "min",
    "small",
    "std",
    "tiny",
]
X = pd.read_csv(os.path.join(dir, "data/eig_bpp_instances_only_features.csv"))
X = X[features]


def experimental_work_test(transformer: KerasNN, *args):
    predicted = transformer.predict(X)
    loss = mean_squared_error(X, predicted)
    return loss


def test_hyper_cmaes_bpp():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )
    cma_es = NNTuner(
        dimension=dimension,
        direction="maximise",
        transformer=transformer,
        generations=5,
        eval_fn=experimental_work_test,
        lambda_=5,
    )
    best_nn_weights, population, logbook = cma_es()
    assert len(best_nn_weights) == dimension
    assert len(population) == cma_es._lambda
    assert len(logbook) == 5


def test_hyper_cmaes_bpp_maximises():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )
    cma_es = NNTuner(
        dimension=dimension,
        direction="minimise",
        transformer=transformer,
        generations=5,
        eval_fn=experimental_work_test,
        lambda_=5,
    )
    best_nn_weights, population, logbook = cma_es()
    assert len(best_nn_weights) == dimension
    assert len(population) == cma_es._lambda
    assert len(logbook) == 5


def test_hyper_cmaes_raises():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )

    # Raises because we do not specify any valid direction
    with pytest.raises(AttributeError):
        cma_es = NNTuner(
            dimension=dimension,
            generations=5,
            eval_fn=experimental_work_test,
            transformer=transformer,
            direction="random_direction",
        )

    # Raises because we do not specify any transformer
    with pytest.raises(AttributeError):
        cma_es = NNTuner(
            transformer=None,
            dimension=dimension,
            direction="maximise",
            generations=5,
            eval_fn=experimental_work_test,
        )

    # Raises because we do not specify any eval_fn
    with pytest.raises(AttributeError):
        cma_es = NNTuner(
            dimension=dimension,
            direction="maximise",
            generations=5,
            transformer=transformer,
            eval_fn=None,
        )

    # Raises because we n_jobs < 1
    with pytest.raises(AttributeError):
        cma_es = NNTuner(
            dimension=dimension,
            direction="maximise",
            generations=5,
            transformer=transformer,
            eval_fn=experimental_work_test,
            n_jobs=-1,
        )

    cma_es = NNTuner(
        dimension=dimension,
        direction="maximise",
        generations=5,
        transformer=transformer,
        eval_fn=experimental_work_test,
        n_jobs=4,
    )
    assert cma_es.n_processors == 4
    assert cma_es.pool is not None
