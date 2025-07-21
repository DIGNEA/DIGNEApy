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

import pytest


def test_imports_utils():
    import digneapy.utils as utils

    assert utils is not None


def test_import_generators():
    import digneapy.generators as generators

    assert generators is not None


def test_import_solvers():
    import digneapy.solvers as solvers

    assert solvers is not None


def test_import_visualize():
    import digneapy.visualize as visualize

    assert visualize is not None
