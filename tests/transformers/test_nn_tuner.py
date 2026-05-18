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

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available on this platform")
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

from digneapy.transformers.neural import NNEncoder
from digneapy.transformers.tuner import Tuner

DIR = Path(__file__).parent

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
X = pd.read_csv(DIR / "data/eig_bpp_instances_only_features.csv")
X = X[features].values


def experimental_work_test(transformer: NNEncoder, *args):
    predicted = transformer.predict(X)
    loss = mean_squared_error(X, predicted)
    return loss


def test_hyper_cmaes_bpp():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = NNEncoder(
        expected_filename,
        input_shape=[11],
        shape=shapes,
        activations=activations,
    )
    cma_es = Tuner(
        dimension=dimension,
        evaluations=5,
        lambda_=5,
        ranges=(0.0, 0.0),
    )
    best_nn_weights, population, logbook = cma_es(None)
    assert len(best_nn_weights) == dimension
    assert len(population) == cma_es._lambda
    assert len(logbook) == 5


def test_hyper_cmaes_bpp_maximises():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = NNEncoder(
        expected_filename,
        input_shape=[11],
        shape=shapes,
        activations=activations,
    )
    cma_es = Tuner(
        dimension=dimension,
        evaluations=5,
        lambda_=5,
        ranges=(0.0, 0.0),
    )
    best_nn_weights, population, logbook = cma_es(None)
    assert len(best_nn_weights) == dimension
    assert len(population) == cma_es._lambda
    assert len(logbook) == 5


def test_hyper_cmaes_raises():
    dimension = 291
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = NNEncoder(
        expected_filename,
        input_shape=[11],
        shape=shapes,
        activations=activations,
    )

    # Raises because we n_jobs < 1
    with pytest.raises(ValueError):
        _ = Tuner(
            dimension=dimension,
            evaluations=5,
            workers=-1,
            ranges=(0.0, 0.0),
        )
