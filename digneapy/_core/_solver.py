#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _solver.py
@Time    :   2026/05/15 11:03:12
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from abc import abstractmethod
from typing import Protocol

from ._problem import Problem
from ._protocols import RandGen
from ._solution import Solution


class Solver(RandGen, Protocol):
    """Solver is any callable type that receives a Problem (P)
    as argument and returns a list of Solution objects
    """

    @abstractmethod
    def __call__(self, problem: Problem, *args, **kwargs) -> list[Solution]:
        """Solves a optimisation problem

        Args:
            problem (Problem): Any optimisation problem or callablle that receives a Sequence and returns a Tuple[float]

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            List[Solution]: Returns a sequence of olutions
        """
        msg = "__call__ method not implemented in Solver"
        raise NotImplementedError(msg)
