#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sample_bpp.py
@Time    :   2024/06/18 11:13:32
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import numpy as np

from digneapy.domains import BPP, BPPDomain


def bpp_sample():
    dimension = 100
    domain = BPPDomain(dimension, capacity_approach="fixed", max_capacity=100)
    instance = domain.generate_instance()
    features = domain.extract_features(instance)

    print(list(instance))
    print(features)

    capacity = instance._variables[0]
    items = np.asarray(instance._variables[1:])
    items_norm = items / capacity

    assert isinstance(features, tuple)
    expected_f = (
        capacity,
        np.mean(items_norm),
        np.std(items_norm),
        np.median(items_norm),
        np.max(items_norm),
        np.min(items_norm),
    )
    assert expected_f == features[:5]
    assert all(float(f) for f in features[5:])

    domain.capacity_approach = "evolved"
    features = domain.extract_features(instance)
    assert features[0] == instance[0]

    domain.capacity_approach = "percentage"
    features = domain.extract_features(instance)
    expected_q = int(np.sum(instance._variables[1:]) * 0.8)
    assert features[0] == expected_q


if __name__ == "__main__":
    bpp_sample()
