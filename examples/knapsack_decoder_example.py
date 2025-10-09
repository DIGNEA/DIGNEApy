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

from digneapy.transformers.autoencoders import KPDecoder, KPEncoder
from digneapy.domains import KnapsackDomain, Knapsack
from typing import Iterable
import numpy as np
import time
from tqdm import tqdm

if __name__ == "__main__":
    N_INSTANCES = 1_000_000
    decoder = KPDecoder(scale_method="sample")

    encodings = np.random.default_rng().uniform(low=-4, high=4, size=(N_INSTANCES, 2))
    # print(encodings)
    start = time.perf_counter()
    decoded_instances = decoder(encodings)
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed} seconds to KPDecoder to decode 1M instances")
    print(decoded_instances)

    print("Now creating instances")
    domain = KnapsackDomain()
    start = time.perf_counter()
    instances = np.asarray(
        [np.array(domain.generate_instances()) for _ in range(N_INSTANCES)]
    )
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed} seconds to create 1M instances seq")
    print(instances)
    start = time.perf_counter()
    instances = domain.generate_n_instances(N_INSTANCES)
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed} to create 1M instance numpy")
    print(instances)

    encoder = KPEncoder()
    start = time.perf_counter()
    encodings = encoder(instances)
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed} seconds to KPEncoder to encode 1M KP instances")
    print(encodings)
