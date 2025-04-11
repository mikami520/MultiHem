#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 23:26:08
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-04 19:56:43
FilePath     : /Downloads/MultiHem/src/loss.py
Description  : Loss Functions of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch import Tensor
import monai.networks.blocks.warp as warp_module
from monai import losses


__all__ = [
    "DiceLoss",
    "SegLoss",
    "RegLoss",
    "ClassLoss",
]


class DiceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.dice = losses.DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            reduction=reduction,
        )

    def forward(self, pred, target):
        loss = self.dice(pred, target)
        return loss


class SegLoss(nn.Module):
    def __init__(self, lambda_super, lambda_ana, reduction="mean"):
        super(SegLoss, self).__init__()
        self.dice = losses.DiceLoss(
            include_background=False,
            to_onehot_y=False,
            softmax=False,
            reduction=reduction,
        )
        # self.lambda_super = lambda_super
        # self.lambda_ana = lambda_ana
        # self.warp = warp_module.Warp(mode="nearest", padding_mode="border")

    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        seg1: torch.Tensor | None,
        seg2: torch.Tensor | None,
        disp: torch.Tensor,
    ) -> torch.Tensor:
        if seg1 is not None and seg2 is not None:
            loss_super = self.dice(pred1, seg1) + self.dice(pred2, seg2)
        elif seg1 is not None:
            loss_super = self.dice(pred1, seg1)
            seg2 = pred2
        else:
            assert seg2 is not None
            loss_super = self.dice(pred2, seg2)
            seg1 = pred1

        warped_seg = self.warp(seg2, disp)
        loss_anatomy = (
            self.dice(warped_seg, seg1) if seg1 is not None or seg2 is not None else 0.0
        )
        seg_loss = self.lambda_super * loss_super + self.lambda_ana * loss_anatomy

        return seg_loss


class RegLoss(nn.Module):
    def __init__(
        self, lambda_sim, lambda_penal, lambda_ana, reduction="mean", num_classes=2
    ):
        super(RegLoss, self).__init__()
        self.sim = losses.LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=7,
            kernel_type="rectangular",
            reduction=reduction,
        )
        self.penalty = losses.BendingEnergyLoss(normalize=True, reduction=reduction)
        self.dice = losses.DiceLoss(
            include_background=False,
            to_onehot_y=False,
            softmax=False,
            reduction=reduction,
        )
        self.lambda_sim = lambda_sim
        self.lambda_penal = lambda_penal
        self.lambda_ana = lambda_ana
        self.num_classes = num_classes
        self.warp = warp_module.Warp(mode="bilinear", padding_mode="border")

    def forward(
        self,
        img_pair: torch.Tensor,
        disp: torch.Tensor,
        seg1: torch.Tensor,
        seg2: torch.Tensor,
    ):
        moving_img = img_pair[:, [1], :, :, :]
        target_img = img_pair[:, [0], :, :, :]
        warped_img = self.warp(moving_img, disp)
        warped_seg = self.warp(seg2, disp)
        sim_loss = self.sim(warped_img, target_img)
        penalty_loss = self.penalty(disp)
        dice_loss = self.dice(warped_seg, seg1)
        reg_loss = (
            self.lambda_sim * sim_loss
            + self.lambda_penal * penalty_loss
            + self.lambda_ana * dice_loss
        )
        # print(
        #     f"sim_loss: {sim_loss.item():.4f}, penalty_loss: {penalty_loss.item():.4f}, dice_loss: {dice_loss.item():.4f}"
        # )
        return reg_loss


class ClassLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduction="mean",
        label_smoothing: float = 0,
    ):
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            size_average=size_average,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target.long())
