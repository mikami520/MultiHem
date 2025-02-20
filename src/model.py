#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:26:14
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-19 21:04:55
FilePath     : /MultiHem/src/model.py
Description  : Model Architectures of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn as nn
from src.backbone import SegNet, RegNet, Fusion, Classifier

__all__ = ["MultiHem"]


class MultiHem(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MultiHem, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.model = None

    def forward(self, x):
        pass
