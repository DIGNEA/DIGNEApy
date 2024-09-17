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


def __getattr__(name):
    if name == "clock":
        from ._timers import clock as clock

        return clock

    if name == "plot_generator_logbook":
        from ._plots import plot_generator_logbook as pgl

        return pgl

    if name == "plot_map_elites_logbook":
        from ._plots import plot_map_elites_logbook as pml

        return pml
