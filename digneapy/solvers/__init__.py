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

from . import bpp as bpp
from . import evo as evo
from . import kp as kp
from . import pisinger

__all__ = ["evo", "bpp", "kp", "pisinger"]


# def __getattr__(attr_name):
#     ret_mod = None

#     if ret_mod is None:
#         raise AttributeError(f"module 'digneapy.solvers' has no attribute {attr_name}")
#     return ret_mod
