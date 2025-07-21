#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/14 13:45:31
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._timers import clock, clock_to_file
from .save_data import save_results_to_files
from .serializer import to_json

__all__ = ["clock", "clock_to_file", "save_results_to_files", "to_json"]
