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

from functools import wraps
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from ..transformers import Transformer
from ._domain import Domain
from ._instance import Instance

type DescriptorKey = str


class DescriptorFn(Protocol):
    """Defines the Protocol that all descriptable functions must follow"""

    def __call__(
        self,
        population: np.ndarray | Sequence[Instance],
        scores: Optional[np.ndarray],
        domain: Optional[Domain],
        *args,
        **kwargs,
    ) -> np.ndarray: ...


descriptors_registry: dict[DescriptorKey, DescriptorFn] = {}


def register_descriptor(key: str) -> Callable[[DescriptorFn], DescriptorFn]:
    """Registers a new descriptor strategy

    Args:
        key (str): Key to store the strategy

    Returns:
        Callable[[DescriptorFn], DescriptorFn]: Newly registered strategy
    """

    def decorator(fn: DescriptorFn) -> DescriptorFn:
        @wraps(fn)
        def wrapper(
            population: np.ndarray | Sequence[Instance],
            scores: Optional[np.ndarray],
            domain: Optional[Domain],
            *args,
            **kwargs,
        ) -> np.ndarray:
            return fn(population, scores, domain, *args, **kwargs)

        descriptors_registry[key] = wrapper
        return wrapper

    return decorator


class DescriptorPipeline:
    """Pipeline to transform descriptors with several models"""

    def __init__(self, key: DescriptorKey, *transformers: Transformer):
        if not isinstance(key, str) or key not in descriptors_registry:
            raise KeyError(
                f"Unknown descriptor key {key}. Registered keys are: {descriptors_registry.keys()}"
            )
        if any(not isinstance(t, Transformer) for t in transformers):
            raise TypeError(
                f"All transformers must implement the Transformer Protocol. Got: {transformers}"
            )
        self._key = key
        self._transformers = tuple(transformers)

    def __call__(
        self,
        population: np.ndarray | Sequence[Instance],
        scores: Optional[np.ndarray] = None,
        domain: Optional[Domain] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        descriptors = descriptors_registry[self._key](
            population, scores, domain, *args, **kwargs
        )
        for transfomer in self._transformers:
            descriptors = transfomer(descriptors)
        return descriptors

    def __repr__(self) -> str:
        steps = [self._key] + [
            getattr(t, "__name__", repr(t)) for t in self._transformers
        ]
        return f"DescriptorPipeline({' -> '.join(steps)})"


@register_descriptor(key="features")
def describe_features(
    population: np.ndarray | Sequence[Instance],
    scores: Optional[np.ndarray],
    domain: Optional[Domain],
    *args,
    **kwargs,
) -> np.ndarray:
    if domain is None:
        raise ValueError("Domain cannot be None when the key is features")
    descriptors = np.empty(len(population))
    descriptors = domain.extract_features(population)
    return descriptors


@register_descriptor(key="performance")
def describe_performance(
    population: np.ndarray | Sequence[Instance],
    scores: Optional[np.ndarray],
    domain: Optional[Domain],
    *args,
    **kwargs,
) -> np.ndarray:
    if scores is None:
        raise ValueError("Scores cannot be None when the key is performance")
    return np.mean(scores, axis=2)


@register_descriptor(key="instance")
def describe_instance(
    population: np.ndarray | Sequence[Instance],
    scores: Optional[np.ndarray],
    domain: Optional[Domain],
    *args,
    **kwargs,
) -> np.ndarray:
    return np.asarray([*population])
