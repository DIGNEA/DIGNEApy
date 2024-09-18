#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2024/09/18 11:03:14
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from ._evo_generator_evolution_plot import ea_generator_evolution_plot
from ._map_elites_evolution_plot import map_elites_evolution_plot

__all__ = ["ea_generator_evolution_plot", "map_elites_evolution_plot"]
