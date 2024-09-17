#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/06/10 12:19:05
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._solver import ParEAKP

__all__ = ["ParEAKP"]


def __getattr__(name):
    if name == "ParEAKP":
        from ._solver import ParEAKP as ParEAKP

        return ParEAKP
