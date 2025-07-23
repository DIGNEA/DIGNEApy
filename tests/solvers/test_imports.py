#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_imports.py
@Time    :   2025/04/08 11:39:46
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""


def test_imports_evo():
    import digneapy.solvers.evolutionary as evo

    assert evo is not None


def test_imports_pisinger():
    import digneapy.solvers.pisinger as pisinger

    assert pisinger is not None
