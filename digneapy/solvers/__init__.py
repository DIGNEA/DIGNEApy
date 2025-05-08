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

from . import bpp, tsp, _kp, _tsp_opt
from .bpp import best_fit, first_fit, next_fit, worst_fit
from ._kp import default_kp, map_kp, miw_kp, mpw_kp
from .tsp import greedy, nneighbour, three_opt
from ._tsp_opt import two_opt

__all__ = list(
    set(bpp.__all__) | set(_kp.__all__) | set(tsp.__all__) | set(_tsp_opt.__all__)
)

__solvers_modules = {"evolutionary", "pisinger"}


# Lazy import function
def __getattr__(attr_name):
    import importlib
    import sys

    if attr_name in __solvers_modules:
        full_name = f"digneapy.solvers.{attr_name}"
        submodule = importlib.import_module(full_name)
        sys.modules[full_name] = submodule
        return submodule

    else:
        raise ImportError(f"module digneapy.solvers has no attribute {attr_name}")
