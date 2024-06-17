#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _timers.py
@Time    :   2024/06/14 13:45:40
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

import time
from collections.abc import Iterable


def clock(func):
    def clocked(*args):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        name = func.__name__
        args_str = ",  ".join(repr(arg) for arg in args)
        result_str = (
            "\n" + "\n - ".join(repr(r) for r in result)
            if isinstance(result, Iterable)
            else repr(result)
        )
        print(f"[{elapsed:0.8f}s] {name}({args_str}) -> {result_str}")
        return result

    return clocked
