#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2023/11/02 12:06:59
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2023, Alejandro Marrero
@Desc    :   None
"""

from . import _kp, _tsp_opt, bpp, evolutionary, tsp, utils
from ._kp import default_kp, map_kp, miw_kp, mpw_kp
from ._tsp_opt import two_opt
from .bpp import best_fit, first_fit, next_fit, worst_fit
from .evolutionary import EA
from .tsp import greedy, nneighbour, three_opt
from .utils import shuffle_and_run_for_knapsack

__all__ = list(
    set(bpp.__all__)
    | set(_kp.__all__)
    | set(tsp.__all__)
    | set(evolutionary.__all__)
    | set(_tsp_opt.__all__)
    | set(utils.__all__)
)
