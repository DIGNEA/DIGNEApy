#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   example.py
@Time    :   2026/05/15 18:47:47
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from functools import wraps
from typing import Callable, Sequence

import numpy as np

type Instance = np.ndarray | Sequence
type DescriptorFn = Callable[[Instance], np.ndarray]
descriptors: dict[str, DescriptorFn] = {}


def register_descriptor(key: str) -> Callable[[DescriptorFn], DescriptorFn]:
    def decorator(fn: DescriptorFn) -> DescriptorFn:
        @wraps(fn)
        def wrapper(instance: Instance) -> np.ndarray:
            return fn(instance)

        descriptors[key] = wrapper
        return wrapper

    return decorator


@register_descriptor(key="features")
def describe_features(instances: Instance):
    print("Registers features")
    return np.zeros(10)


if __name__ == "__main__":
    for d in descriptors.keys():
        print(d, descriptors[d].__name__, descriptors[d]([]))
