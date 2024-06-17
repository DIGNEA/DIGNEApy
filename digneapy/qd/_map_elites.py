#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _map_elites.py
@Time    :   2024/06/17 10:12:09
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""

from digneapy.archives import GridArchive
from digneapy.operators import mutation


class MapElites:
    def __init__(self, archive: GridArchive, mutation: mutation.Mutation, bounds):
        self._archive = archive
        self._mutation = mutation

    @property
    def archive(self):
        return self._archive

    def __str__(self):
        return "MapElites()"

    def __repr__(self) -> str:
        return "MapElites<>"
