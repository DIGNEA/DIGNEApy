#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   pisinger.py
@Time    :   2024/09/17 12:40:27
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


def __getattr__(attr_name):
    from digneapy.solvers import _pisinger

    ret_module = getattr(_pisinger, attr_name, None)
    if ret_module is None:
        raise ImportError(
            f"module 'digneapy.solvers.pisinger' has no attribute {attr_name}"
        )
    return ret_module
