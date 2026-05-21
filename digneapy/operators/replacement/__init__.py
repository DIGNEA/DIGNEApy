#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2026/05/21 15:23:31
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from .base import Replacement
from .elitist import Elitist
from .generational import Generational
from .greedy import GreedyReplacement

__all__ = ["Replacement", "Generational", "GreedyReplacement", "Elitist"]
