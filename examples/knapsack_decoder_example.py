#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   knapsack_decoder_example.py
@Time    :   2025/09/30 15:03:31
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from digneapy.transformers.autoencoders import KPDecoder
from digneapy.domains import KnapsackDomain, Knapsack
from typing import Iterable
import numpy as np
import time


if __name__ == "__main__":
    N_INSTANCES = 1_000_000
    encoder = KPDecoder(scale_method="sample")

    encodings = np.random.default_rng().uniform(low=-4, high=4, size=(N_INSTANCES, 2))
    # print(encodings)
    start = time.perf_counter()
    decoded_instances = encoder(encodings)
    elapsed = time.perf_counter() - start
    print(f"Took {elapsed} seconds to decode 1M instances")
    print(decoded_instances)
