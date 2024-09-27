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

from . import bpp, kp
from .bpp import best_fit, first_fit, next_fit, worst_fit
from .kp import default_kp, map_kp, miw_kp, mpw_kp

__all__ = list(set(bpp.__all__) | set(kp.__all__))

__solvers_modules = {"evo", "pisinger"}


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
