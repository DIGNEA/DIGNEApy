#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _run.py
@Time    :   2026/06/25 12:33:42
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from pathlib import Path
from typing import Callable

from digneapy.generators import BaseGenerator, GenerationResult
from digneapy.utils import save_results_to_files

type RunFn = Callable[[BaseGenerator, ...], GenerationResult]


class Run:
    def __init__(self, generator: BaseGenerator, run_name: str, base_dir: Path):
        self._name = run_name
        self._generator = generator
        self._base_dir = base_dir

    def __call__(self) -> GenerationResult:
        result = self._generator()
        save_results_to_files(
            filename_pattern=self._name,
            base_dir=self._base_dir,
            result=result,
            only_instances=True,
        )
        return result
