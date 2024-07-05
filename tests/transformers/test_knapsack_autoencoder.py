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

from digneapy.domains import knapsack
from digneapy.transformers.autoencoders import KPAE

encodings = [2, 8]


@pytest.mark.parametrize("encoding", encodings)
def test_autoencoder(encoding):
    dimension = 1000
    n_instances = 100
    autoencoder = KPAE(encoding=encoding)
    domain = knapsack.KPDomain(dimension=dimension)
    instances = [domain.generate_instance() for _ in range(n_instances)]

    assert isinstance(autoencoder, KPAE)
    encodings = autoencoder.encode(instances)
    assert len(encodings) == n_instances
    assert all(len(enc_i) == encoding for enc_i in encodings)

    decodings = autoencoder.decode(encodings)
    assert len(decodings) == n_instances
    assert all(len(dec_i) == (dimension * 2) + 1 for dec_i in decodings)
