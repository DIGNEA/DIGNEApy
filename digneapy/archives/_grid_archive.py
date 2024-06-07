#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _grid_archive.py
@Time    :   2024/06/07 12:18:10
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._base_archive import Archive
from digneapy.core import Instance
import numpy as np
import operator
import reprlib
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Optional


class GridArchive(Archive):
    def __init__(
        self,
        dimensions: Sequence[int],
        ranges: Sequence[float],
        lr: float,
        instances: Optional[Iterable[Instance]] = None,
    ):
        Archive.__init__(self, instances)
        self._dimensions = tuple(dimensions)
        self._lower_bounds = np.array(ranges[0], dtype=np.float64)
        self._upper_bounds = np.array(ranges[1], dtype=np.float64)
        self._interval_size = self._upper_bounds - self._lower_bounds
