#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/07 14:28:12
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.qd._novelty_search import NS
from digneapy.qd.desc_strategies import (
    features_strategy,
    performance_strategy,
    instance_strategy,
)


__all__ = ["NS", "features_strategy", "performance_strategy", "instance_strategy"]
