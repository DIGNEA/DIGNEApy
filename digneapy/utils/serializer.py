#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   serializer.py
@Time    :   2025/04/02 16:00:58
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

__all__ = ["to_json"]

import json
from types import FunctionType

import numpy as np


def serialize(obj):
    """
    Recursively serialize an object to a dictionary.
    Handles nested objects, lists, and dictionaries.
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj  # Primitive types are directly serializable

    if (
        isinstance(obj, (FunctionType))
        or type(obj).__name__ == "cython_function_or_method"
    ):
        return obj.__name__

    if isinstance(obj, (list, tuple, dict)):
        return [serialize(item) for item in obj]

    # if isinstance(obj, (list, tuple)) or isinstance(obj, dict):
    #     return [
    #         serialize(item) for item in obj
    #     ]  # Serialize each element in the list/tuple

    if isinstance(obj, dict):
        return {
            key: serialize(value) for key, value in obj.items()
        }  # Serialize each key-value pair

    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists

    if hasattr(obj, "__dict__"):  # Handle custom objects
        return {
            key: serialize(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    if hasattr(obj, "__slots__"):
        return {slot: serialize(getattr(obj, slot)) for slot in obj.__slots__}

    return str(obj)  # Fallback: Convert to string for unsupported types


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle complex types like NumPy arrays and custom objects.
    """

    def default(self, obj):
        try:
            return serialize(obj)
        except TypeError:
            return super().default(obj)


def to_json(obj):
    """
    Convert an object to a JSON string.
    """
    return json.dumps(serialize(obj), cls=CustomJSONEncoder, indent=4)
