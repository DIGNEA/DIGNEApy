#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   problem.py
@Time    :   2024/06/07 14:07:55
@Author  :   Alejandro Marrero 
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2024, Alejandro Marrero
@Desc    :   None
"""


from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Tuple, TypeVar


class Problem(ABC):
    def __init__(self, name: str = "DefaultOptProblem", *args, **kwargs):
        self._name = name

    @abstractmethod
    def evaluate(self, individual: Sequence) -> Tuple[float]:
        """Evaluates the candidate individual with the information of the Knapsack

        Args:
            individual (Sequence): Individual to evaluate

        Raises:
            AttributeError: Raises an error if the len(individual) != len(instance) / 2

        Returns:
            Tuple[float]: Profit
        """
        msg = "evaluate method not implemented in OptProblem"
        raise NotImplementedError(msg)

    @abstractmethod
    def __call__(self, individual: Sequence) -> Tuple[float]:
        msg = "__call__ method not implemented in OptProblem"
        raise NotImplementedError(msg)


P = TypeVar("P", bound=Problem)
