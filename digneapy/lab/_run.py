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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from digneapy.generators import GenerationResult
from digneapy.utils import save_results_to_files

type RunFn = Callable[..., GenerationResult]


@dataclass(slots=True)
class RunConfig:
    """Configuration for a single experiment run.

    A run config bundles the callable to execute, the name used for output,
    optional arguments and keyword arguments, and persistence settings for
    saving generated results to disk.
    """

    call_fn: RunFn
    name: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    save_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, result_directory: Optional[Path] = None) -> GenerationResult:
        """Execute the configured callable and optionally persist the result.

        Returns:
            The generation result produced by the wrapped callable.
        """
        result: GenerationResult = self.call_fn(*self.args, **self.kwargs)
        if result_directory is not None:
            save_results_to_files(
                filename_pattern=self.name,
                base_dir=result_directory,
                result=result,
                **self.save_kwargs,
            )
        return result
