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

from digneapy.solvers.evolutionary import EA
from digneapy.solvers.heuristics import default_kp, map_kp, miw_kp, mpw_kp

__all__ = ["EA", "default_kp", "map_kp", "miw_kp", "mpw_kp"]
