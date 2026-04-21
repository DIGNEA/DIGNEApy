#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   descriptors.py
@Time    :   2026/03/25 11:57:22
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Literal, Optional, Protocol, Sequence, Tuple

import numpy as np

from .._core import Domain, Instance
from ..transformers import SupportsTransform

DESCRIPTORS = Literal["features", "performance", "instance"]


class Descriptable(Protocol):
    """Defines the Protocol that all descriptable functions must follow"""

    def __call__(
        self,
        population: np.ndarray,
        key: DESCRIPTORS,
        scores: Optional[np.ndarray] = None,
        domain: Optional[Domain] = None,
        transformer: Optional[SupportsTransform] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...


def describe(
    population: np.ndarray | Sequence[Instance],
    key: DESCRIPTORS,
    scores: Optional[np.ndarray] = None,
    domain: Optional[Domain] = None,
    transformer: Optional[SupportsTransform] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Updates the descriptors of the population of instances

    Args:
        population (np.ndarray): Population of instances to describe
        key (Literal[&quot;features&quot;, &quot;performance&quot;, &quot;instance&quot;]): Type of descriptor to extract
        scores (Optional[np.ndarray], optional): Scores of the solvers. Defaults to None.
        domain (Optional[Domain], optional): Domain to extract the features if needed. Defaults to None.
        transformer (Optional[SupportsTransform], optional): Transformer to transform the descriptor after extracted. Defaults to None.

    Raises:
        ValueError: If the key is not features, performance or instance
        ValueError: If key is features and domain is None
        ValueError: If key is performance and scores is None

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Descriptors and features if necessary
    """
    if key not in ("features", "performance", "instance"):
        raise ValueError("Expected key to be features, performance or instance")

    descriptors = np.empty(len(population))
    features = None

    if key == "features":
        if domain is None:
            raise ValueError("Domain cannot be None when the key is features")
        descriptors = domain.extract_features(population)
        features = descriptors.copy()
    elif key == "performance":
        if scores is None:
            raise ValueError("Scores cannot be None when the key is performance")
        descriptors = np.mean(scores, axis=2)
    elif key == "instance":
        descriptors = np.asarray([*population])

    if transformer is not None:
        descriptors = transformer(descriptors)

    return (descriptors, features)
