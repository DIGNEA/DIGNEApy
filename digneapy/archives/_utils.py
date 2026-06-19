#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/06/10 11:02:26
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from collections.abc import Sequence

from digneapy.core import Instance


def check_valid_instance_batch(instances: Sequence[Instance]) -> bool:
    """Checks that all objects are valid Instances

    Args:
        instances (Sequence[Instance]): Sequence of object of the Instance class

    Returns:
        bool: If all of the objects inside the instances parameter are valid
    """
    return all(isinstance(x, Instance) for x in instances)


def check_valid_shapes(instances: Sequence[Instance], *args) -> bool:
    """Check that all components have the same length of the instances

    Args:
        instances (Sequence[Instance]): Collection of instances to compare the length to.

    Returns:
        bool: True if all the other components have the same length. False otherwise.
    """
    expected = len(instances)
    try:
        return all(len(component) == expected for component in args)
    except Exception:
        return False
