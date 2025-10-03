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
import numpy as np
from digneapy.domains import KnapsackDomain
from digneapy.transformers.autoencoders import KPEncoder, KPDecoder

scale_methods = ("learnt", "sample")


@pytest.mark.parametrize("scale_method", scale_methods)
def test_KPDecoder_works_with_scale_method(scale_method):
    N_INSTANCES = 1_000_000
    decoder = KPDecoder(scale_method=scale_method)
    encodings = np.random.default_rng().normal(0, 0.1, size=(N_INSTANCES, 2))
    decodings = decoder(encodings)
    assert encodings.shape[0] == decodings.shape[0]
    assert encodings.shape[1] == 2
    assert decodings.shape[1] == decoder.output_dimension


def test_KPEncoder_works_with_Knapsack_instances():
    N_INSTANCES = 1_000_000
    encoder = KPEncoder()
    domain = KnapsackDomain()
    instances = np.asarray(
        [np.array(domain.generate_instance()) for _ in range(N_INSTANCES)]
    )
    assert instances.shape == (N_INSTANCES, encoder._expected_input_dim)
    encodings = encoder(instances)
    assert encodings.shape[0] == instances.shape[0]
    assert encodings.shape[1] == encoder.latent_dimension
