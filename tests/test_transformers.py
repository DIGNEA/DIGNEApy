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
from __future__ import unicode_literals

import os
import pytest
import numpy as np
import pandas as pd
from digneapy.transformers import Transformer, KerasNN, TorchNN
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


def test_default_transformer():
    t = Transformer(name="Default")
    assert t._name == "Default"

    with pytest.raises(Exception):
        t.train(list())

    with pytest.raises(Exception):
        t.predict(list())

    with pytest.raises(Exception):
        t.save()

    with pytest.raises(Exception):
        t(list())


def test_KerasNN_transformer_raises():
    shapes = (11, 5, 2)
    activations = ("relu", "relu", None)
    expected_filename = "nn_transformer_bpp"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )

    assert transformer._name == expected_filename + ".keras"
    assert type(transformer.__str__()) == str
    assert type(transformer.__repr__()) == str

    with pytest.raises(Exception):
        # Raises exception because
        # len(activations) != len(shape)
        transformer = KerasNN(
            expected_filename,
            input_shape=[11],
            shape=shapes,
            activations=activations[1:],
        )

    with pytest.raises(Exception):
        # Raises exception because X is empty
        transformer.train(list())

    with pytest.raises(Exception):
        # Raises exception because X is empty
        transformer.predict(list())

    with pytest.raises(Exception):
        # Raises exception because X is empty when calling __call__
        transformer(list())


def test_KerasNN_transformer_bpp():
    shapes = (11, 5, 2)
    activations = ("relu", "relu", None)
    expected_filename = "nn_transformer_bpp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )

    weights = np.random.random_sample(size=204)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    x = np.array([np.random.sample(size=11) for _ in range(100)])
    predicted = transformer.predict(x)
    assert len(predicted) == 100
    assert all(len(x_i) == 2 for x_i in predicted)

    transformer.save()
    assert os.path.exists(expected_filename) == True
    os.remove(expected_filename)

    # Now saving using a different filename
    new_filename = "random_transformer.keras"
    transformer.save(filename=new_filename)
    assert os.path.exists(new_filename) == True
    os.remove(new_filename)


def test_KerasNN_transformer_kp():
    shapes = (8, 4, 2)
    activations = ("relu", "relu", None)
    expected_filename = "nn_transformer_kp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[8], shape=shapes, activations=activations
    )

    weights = np.random.random_sample(size=118)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    x = np.array([np.random.sample(size=8) for _ in range(100)])
    predicted = transformer.predict(x)
    assert len(predicted) == 100
    assert all(len(x_i) == 2 for x_i in predicted)

    transformer.save()
    assert os.path.exists(os.path.join(os.path.curdir, expected_filename))
    os.remove(os.path.join(os.path.curdir, expected_filename))


def test_KerasNN_reduced_transformer_kp():
    shapes = (4, 2)
    activations = ("relu", None)
    expected_filename = "nn_transformer_kp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[8], shape=shapes, activations=activations
    )

    weights = np.random.random_sample(size=46)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    x = np.array([np.random.sample(size=8) for _ in range(100)])
    predicted = transformer.predict(x)
    assert len(predicted) == 100
    assert all(len(x_i) == 2 for x_i in predicted)

    transformer.save()
    assert os.path.exists(os.path.join(os.path.curdir, expected_filename))
    os.remove(os.path.join(os.path.curdir, expected_filename))


def test_KerasNN_autoencoder_bpp():
    shapes = (11, 5, 2, 2, 5, 11)
    activations = ("relu", "relu", None, "relu", "relu", None)
    expected_filename = "nn_autoencoder_bpp.keras"
    transformer = KerasNN(
        expected_filename, input_shape=[11], shape=shapes, activations=activations
    )

    weights = np.random.random_sample(size=291)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)


###########################################################################################################
##########################################################################################################


def test_TorchNN_transformer_raises():
    expected_filename = "nn_transformer_bpp"
    transformer = TorchNN(
        expected_filename,
        input_size=11,
        output_size=2,
        shape=(5,),
    )

    assert transformer._name == expected_filename + ".torch"
    assert type(transformer.__str__()) == str
    assert type(transformer.__repr__()) == str

    with pytest.raises(Exception):
        # Raises exception because X is empty
        transformer.train(list())

    with pytest.raises(Exception):
        # Raises exception because X is empty
        transformer.predict(list())

    with pytest.raises(Exception):
        # Raises exception because X is empty when calling __call__
        transformer(list())


def test_TorchNN_transformer_bpp():
    expected_filename = "nn_transformer_bpp.torch"
    transformer = TorchNN(
        expected_filename,
        input_size=11,
        shape=(5,),
        output_size=2,
    )

    weights = np.random.random_sample(size=72)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    x = np.array([np.random.sample(size=11) for _ in range(100)])
    predicted = transformer.predict(x)
    assert len(predicted) == 100
    assert all(len(x_i) == 2 for x_i in predicted)

    transformer.save()
    assert os.path.exists(expected_filename) == True
    os.remove(expected_filename)

    # Now saving using a different filename
    new_filename = "random_transformer.torch"
    transformer.save(filename=new_filename)
    assert os.path.exists(new_filename) == True
    os.remove(new_filename)


def test_TorchNN_transformer_kp():
    expected_filename = "nn_transformer_kp.torch"
    transformer = TorchNN(
        expected_filename,
        input_size=8,
        output_size=2,
        shape=(4,),
    )

    weights = np.random.random_sample(size=46)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)

    x = np.array([np.random.sample(size=8) for _ in range(100)])
    predicted = transformer.predict(x)
    assert len(predicted) == 100
    assert all(len(x_i) == 2 for x_i in predicted)

    transformer.save()
    assert os.path.exists(os.path.join(os.path.curdir, expected_filename))
    os.remove(os.path.join(os.path.curdir, expected_filename))


def test_TorchNN_autoencoder_bpp():
    expected_filename = "nn_autoencoder_bpp.torch"
    transformer = TorchNN(
        expected_filename,
        input_size=11,
        output_size=11,
        shape=(5, 2, 5),
    )

    weights = np.random.random_sample(size=138)
    assert transformer is not None
    assert transformer.update_weights(weights) == True
    with pytest.raises(Exception):
        weights = np.random.random_sample(size=1000)
        transformer.update_weights(weights)


def experimental_work_test(transformer: KerasNN, *args):
    predicted = transformer.predict(X)
    loss = mean_squared_error(X, predicted)
    return loss
