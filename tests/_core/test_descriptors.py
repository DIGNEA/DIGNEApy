#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_descriptors.py
@Time    :   2026/06/16 14:14:44
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

import pytest

from digneapy import DescriptorPipeline


@pytest.mark.parametrize("descriptor", ("features", "performance", "instance"))
def test_descriptor_pipeline_can_be_created(descriptor):
    pipeline = DescriptorPipeline(key=descriptor)
    assert isinstance(pipeline, DescriptorPipeline)
    assert pipeline._key == descriptor


@pytest.mark.parametrize("descriptor", ("unknown", list(), None))
def test_descriptor_pipeline_raises_if_wrong_key(descriptor):
    with pytest.raises(KeyError):
        _ = DescriptorPipeline(key=descriptor)
