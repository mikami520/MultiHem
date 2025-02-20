#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:34:17
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-19 21:05:28
FilePath     : /MultiHem/main.py
Description  : Main Function of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
from src import SegNet, RegNet, Fusion, Classifier


def main():
    seg_encode_layers = [16, 48, 96, 192, 384]
    seg_decode_layers = [64, 32, 16, 4]
    reg_encode_layers = [16, 32, 32, 32, 32]
    reg_decode_layers = [32, 32, 32, 32, 16, 16]
    stride = [2, 2, 2, 2]
    seg = SegNet(
        in_channel=3,
        out_channel=1,
        encode_layers=seg_encode_layers,
        decode_layers=seg_decode_layers,
        stride=stride,
    )
    reg = RegNet(
        in_channel=2,
        out_channel=3,
        encode_layers=reg_encode_layers,
        decode_layers=reg_decode_layers,
        stride=stride,
    )
    fusion = Fusion(seg_encode_layers, reg_encode_layers, stride)
    classifier = Classifier(128, 10)
    tensor_seg = torch.randn(2, 3, 64, 64, 64)
    tensor_reg = torch.randn(2, 2, 64, 64, 64)
    output_seg, encoder_feat_seg = seg(tensor_seg)
    output_reg, encoder_feat_reg = reg(tensor_reg)
    output_fusion = fusion(encoder_feat_seg, encoder_feat_reg)
    output_class = classifier(output_fusion)
    print(output_seg.shape, output_reg.shape, output_fusion.shape, output_class.shape)


if __name__ == "__main__":
    main()
