#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_knapsack_autoencoder.py
@Time    :   2024/06/19 12:58:23
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy.domains import kp
from digneapy.transformers.autoencoders import KPEncoder

encoders = (50, 100, 250, 500, 100, "var_2d", "var_8d", "var_best")


@pytest.mark.parametrize("encoder", encoders)
def test_autoencoder(encoder):
    dimension = encoder if isinstance(encoder, int) else 1000
    n_instances = 100
    autoencoder = KPEncoder(encoder=encoder)
    domain = kp.KnapsackDomain(dimension=dimension)
    instances = [domain.generate_instance() for _ in range(n_instances)]

    assert isinstance(autoencoder, KPEncoder)
    encodings = autoencoder.encode(instances)
    assert len(encodings) == n_instances
    if isinstance(encoder, int):
        assert all(len(enc_i) == 2 for enc_i in encodings)
