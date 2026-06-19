#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   typing.py
@Time    :   2026/06/01 15:08:55
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Callable
from typing import TypeVar

import numpy as np

from digneapy.core._instance import Instance
from digneapy.core._solution import Solution

Ind = int | np.integer
Float = float | np.floating

"""
Individual Type in Digneapy to represent Solution and Instances in methods that can be used with both
"""
IndType = TypeVar("IndType", Instance, Solution)

"""
Performance Function type. From any sequence it calculates the performance score.
Returns:
    float: Performance score
"""
PerformanceFn = Callable[[np.ndarray], np.ndarray]
