#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/07 14:21:26
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["EIG", "MElitGen"]


def __getattr__(name):
    if name == "EIG":
        from digneapy.generators._eig import EIG as EIG

        return EIG

    if name == "MElitGen":
        from digneapy.generators._map_elites_gen import MElitGen as MElitGen

        return MElitGen
