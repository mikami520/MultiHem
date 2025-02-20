#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:26:14
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-19 21:14:00
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
        self.SegNet = SegNet(*args, **kwargs)
        self.RegNet = RegNet(*args, **kwargs)
        self.Fusion = Fusion(*args, **kwargs)
        self.Classifier = Classifier(*args, **kwargs)
        self.criterion_seg = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.criterion_class = nn.CrossEntropyLoss()

    def forward(self, x):
        output_seg, encoder_feat_seg = self.SegNet(x)
        output_reg, encoder_feat_reg = self.RegNet(x)
        output_fusion = self.Fusion(encoder_feat_seg, encoder_feat_reg)
        output_class = self.Classifier(output_fusion)
        return output_seg, output_reg, output_fusion, output_class

    def loss(
        self, output_seg, output_reg, output_class, target_seg, target_reg, target_class
    ):
        loss_seg = self.criterion_seg(output_seg, target_seg)
        loss_reg = self.criterion_reg(output_reg, target_reg)
        loss_class = self.criterion_class(output_class, target_class)
        return loss_seg, loss_reg, loss_class

    def metric(
        self, output_seg, output_reg, output_class, target_seg, target_reg, target_class
    ):
        pass
