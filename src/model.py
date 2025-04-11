#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:29:30
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-05 16:09:45
FilePath     : /Downloads/MultiHem/src/model.py
Description  : Backbone Architectures of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

__all__ = [
    "same_padding",
    "SegNet",
    "RegNet",
    "Classifier",
    "BaseSeg",
    "seg_encoder",
    "seg_decoder",
    "seg_decoder_last",
    "DeepSupervisionWrapper",
]

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import copy
import numpy as np
from typing import Sequence, Union, Tuple


def same_padding(
    kernel_size: Sequence[int], dilation: Sequence[int]
) -> Union[Sequence[int], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_num_groups(num_channels, default_num_groups=32):
    num_groups = min(default_num_groups, num_channels)
    while num_groups > 0:
        if num_channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1  # Fallback to 1 if no divisor is found


# Memory-efficient Swish activation function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_input


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_ratio=0.3,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.gn1 = nn.GroupNorm(get_num_groups(out_channels), out_channels, affine=True)
        self.swish = MemoryEfficientSwish()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.gn2 = nn.GroupNorm(get_num_groups(out_channels), out_channels, affine=True)
        self.dropout = nn.Dropout3d(p=dropout_ratio)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.GroupNorm(get_num_groups(out_channels), out_channels, affine=True),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.swish(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.swish(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.swish(out)
        return out


class SqueezeExcitation(nn.Module):
    """
    A Squeeze-and-Excitation block for channel attention in 3D.
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )  # squeeze B x C x D x H x W -> B x C x 1 x 1 x 1
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        b, c = x.shape[:2]
        # Squeeze
        y = self.pool(x).view(b, c)
        # Excitation
        y = self.fc1(y)
        y = self.swish(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view((b, c) + (1,) * (x.ndim - 2))
        return x * y


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=3,
        res_channels=None,
        hidden_dim=64,
        dropout_p=0.3,
        deep_supervision=False,
    ):
        super(Classifier, self).__init__()
        self.res_channels = res_channels if res_channels is not None else in_channels
        self.dropout_p = dropout_p
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.pool_scales = [(1, 1, 1), (2, 2, 2)]
        # Create a small sub-module for each pooling scale
        self.aux_blocks = nn.ModuleList()
        self.scale_blocks = nn.ModuleList()

        for scale in self.pool_scales:
            block = nn.Sequential(
                nn.AdaptiveAvgPool3d(scale),
                ResidualBlock(
                    in_channels,
                    self.res_channels,
                    dropout_ratio=self.dropout_p,
                ),
                SqueezeExcitation(self.res_channels, reduction=16),
            )
            self.scale_blocks.append(block)
            self.aux_blocks.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=self.dropout_p),
                    nn.Linear(
                        self.res_channels * scale[0] * scale[1] * scale[2],
                        hidden_dim,
                    ),
                    nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim, affine=True),
                    MemoryEfficientSwish(),
                    nn.Linear(hidden_dim, num_classes),
                )
            )
        total_flat_dim = sum(
            [
                self.res_channels * scale[0] * scale[1] * scale[2]
                for scale in self.pool_scales
            ]
        )

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc1 = nn.Linear(total_flat_dim, hidden_dim)
        self.gn_fc1 = nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim, affine=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        """
        x: shape (batch_size, in_channels, D, H, W) or (batch_size, in_channels, H, W)
        """
        # 1) Multi-scale pooling + residual + SE
        scale_feats = []
        aux_feats = []
        for i, block in enumerate(self.scale_blocks):
            feat = block(x)  # shape: (B, res_channels, scale_d, scale_h, scale_w)
            feat = feat.view(feat.size(0), -1)  # flatten each scale
            scale_feats.append(feat)
            aux_feats.append(self.aux_blocks[i](feat))
        # 2) Concat across scales
        combined = torch.cat(scale_feats, dim=1)  # shape: (B, total_flat_dim)

        # 3) MLP
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.gn_fc1(x)
        x = self.swish(x)
        x = self.fc2(x)  # shape: (B, num_classes)
        aux_feats.append(x)
        class_outputs = aux_feats[::-1]
        if self.deep_supervision:
            return class_outputs
        else:
            return class_outputs[0]


class SegNet(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 7,
        encode_layers: Sequence[int] = [16, 48, 96, 192, 384],
        decode_layers: Sequence[int] = [64, 32, 16, 4],
        stride: Sequence[int] = [2, 2, 2, 2],
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super(SegNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encode_layers = copy.deepcopy(encode_layers)
        self.encode_layers.insert(0, in_channel)
        self.stride = copy.deepcopy(stride)
        self.stride.insert(0, 1)
        self.dropout = dropout
        self.norm = norm
        self.decode_layers = copy.deepcopy(decode_layers)
        self.encoder_list = nn.ModuleList()
        for i in range(len(self.encode_layers) - 1):
            x = residual(
                self.encode_layers[i],
                self.encode_layers[i + 1],
                (self.stride[i], self.stride[i], self.stride[i]),
                self.dropout,
                self.norm,
            )
            self.encoder_list.append(x)

        self.decoder_list = nn.ModuleList()
        for j in range(len(self.decode_layers)):
            if j == 0:
                decode = seg_decoder(
                    self.encode_layers[len(self.encode_layers) - j - 1],
                    self.decode_layers[j],
                    self.encode_layers[len(self.encode_layers) - j - 2],
                    self.norm,
                    kernel=(1, 1, 1),
                )
            elif j < len(self.decode_layers) - 1:
                decode = seg_decoder(
                    self.decode_layers[j - 1],
                    self.decode_layers[j],
                    self.encode_layers[len(self.encode_layers) - j - 2],
                    self.norm,
                    kernel=(1, 1, 1),
                )
            else:
                decode = seg_decoder_last(
                    self.decode_layers[j - 1],
                    self.decode_layers[j],
                    self.encode_layers[len(self.encode_layers) - j - 2],
                    self.norm,
                )

            self.decoder_list.append(decode)

        self.out0 = nn.Conv3d(
            self.decode_layers[-1], self.out_channel, kernel_size=(1, 1, 1)
        )
        self.out1 = nn.Conv3d(
            self.decode_layers[-2], self.out_channel, kernel_size=(1, 1, 1)
        )
        self.out2 = nn.Conv3d(
            self.decode_layers[-3], self.out_channel, kernel_size=(1, 1, 1)
        )
        self.out3 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        output_list = []
        for _, l in enumerate(self.encoder_list):
            x = l(x)
            output_list.append(x)

        decoder_output = []
        k = len(self.encoder_list)
        for j, ll in enumerate(self.decoder_list):
            if j == 0:
                decode_input = [output_list[k - j - 2], output_list[k - j - 1]]
            else:
                decode_input = [output_list[k - j - 2], decoder_output[j - 1]]

            decode = ll(decode_input)
            decoder_output.append(decode)

        output0 = self.out0(decoder_output[-1])
        output1 = self.out1(decoder_output[-2])
        output2 = self.out2(decoder_output[-3])
        output2_up = F.interpolate(
            output2, scale_factor=2, mode="trilinear", align_corners=True
        )
        output3 = output2_up + output1
        output3_up = F.interpolate(
            output3, scale_factor=2, mode="trilinear", align_corners=True
        )
        output = output3_up + output0
        output = self.out3(output)
        return output, output_list


class RegNet(nn.Module):
    def __init__(
        self,
        in_channel: int = 2,
        out_channel: int = 3,
        encode_layers: Sequence[int] = [16, 32, 32, 32, 32],
        decode_layers: Sequence[int] = [32, 32, 32, 32, 16, 16],
        stride: Sequence[int] = [2, 2, 2, 2],
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super(RegNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encode_layers = copy.deepcopy(encode_layers)
        self.encode_layers.insert(0, self.in_channel)
        self.stride = copy.deepcopy(stride)
        self.stride.insert(0, 1)
        self.dropout = dropout
        self.norm = norm
        self.decode_layers = copy.deepcopy(decode_layers)

        self.encoder_list = nn.ModuleList()
        for i in range(len(self.encode_layers) - 1):
            x = reg_encoder(
                self.encode_layers[i],
                self.encode_layers[i + 1],
                self.norm,
                stride=(self.stride[i], self.stride[i], self.stride[i]),
            )
            self.encoder_list.append(x)

        self.decoder_list = nn.ModuleList()
        for j in range(len(self.decode_layers)):
            if j == 0:
                decode = reg_decoder(
                    self.encode_layers[-1], self.decode_layers[j], self.norm
                )
            elif j < len(self.decode_layers) - 2:
                decode = reg_decoder(
                    self.decode_layers[j - 1]
                    + self.encode_layers[len(self.encode_layers) - j - 1],
                    self.decode_layers[j],
                    self.norm,
                )
            elif j == len(self.decode_layers) - 2:
                decode = reg_encoder(
                    self.decode_layers[j - 1]
                    + self.encode_layers[len(self.encode_layers) - j - 1],
                    self.decode_layers[j],
                    self.norm,
                )
            else:
                decode = reg_encoder(
                    self.decode_layers[j - 1], self.decode_layers[j], self.norm
                )

            self.decoder_list.append(decode)

        self.flow = nn.Conv3d(
            self.decode_layers[-1],
            self.out_channel,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        output_list = []
        x1 = x
        for i, l in enumerate(self.encoder_list):
            x1 = l(x1)
            output_list.append(x1)

        decoder_output = []
        k = len(self.encoder_list)
        for j, ll in enumerate(self.decoder_list):
            if j == 0:
                decode_input = [output_list[k - j - 2], output_list[k - j - 1]]
            elif j < len(self.decode_layers) - 2:
                decode_input = [output_list[k - j - 2], decoder_output[j - 1]]
            else:
                decode_input = decoder_output[j - 1]

            decode = ll(decode_input)
            decoder_output.append(decode)

        disp_field = self.flow(decoder_output[-1])
        return disp_field, output_list


class residual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Sequence[int],
        dropout: float,
        norm: str,
    ) -> None:
        super(residual, self).__init__()
        self.block1 = seg_encoder(in_channels, out_channels, norm, stride=stride)
        self.block2 = seg_encoder(out_channels, out_channels, norm)
        self.drop = nn.Dropout3d(dropout)
        self.block3 = seg_encoder(out_channels, out_channels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.drop(x2)
        x4 = self.block3(x3)
        return x1 + x4


class seg_encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        stride: Sequence[int] = (1, 1, 1),
        kernel: Sequence[int] = (3, 3, 3),
        dilation: Sequence[int] = (1, 1, 1),
    ) -> None:
        super(seg_encoder, self).__init__()
        padding = same_padding(kernel, dilation)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        if norm == "batch":
            self.bn1 = nn.BatchNorm3d(out_channels)
        else:
            self.bn1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class seg_decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        concat_channels: int,
        norm: str,
        stride: Sequence[int] = (2, 2, 2),
        kernel: Sequence[int] = (3, 3, 3),
    ) -> None:
        super(seg_decoder, self).__init__()
        self.conv1 = seg_encoder(in_channels, out_channels, norm)
        self.conv2 = seg_encoder(out_channels + concat_channels, out_channels, norm)
        self.conv3 = seg_encoder(out_channels, out_channels, norm, kernel=kernel)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, encoder_out = x
        up1 = F.interpolate(
            encoder_out,
            scale_factor=self.stride[0],
            mode="trilinear",
            align_corners=True,
        )
        x1 = self.conv1(up1)
        x1 = torch.cat((x1, features), dim=1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        return x1


class reg_encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        stride: Sequence[int] = (1, 1, 1),
        kernel: Sequence[int] = (3, 3, 3),
        dilation: Sequence[int] = (1, 1, 1),
    ) -> None:
        super(reg_encoder, self).__init__()
        padding = same_padding(kernel, dilation)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        if norm == "batch":
            self.bn1 = nn.BatchNorm3d(out_channels)
        else:
            self.bn1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class seg_decoder_last(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        concat_channels: int,
        norm: str,
        stride: Sequence[int] = (2, 2, 2),
    ) -> None:
        super(seg_decoder_last, self).__init__()
        self.conv1 = seg_encoder(in_channels, out_channels, norm)
        self.conv2 = seg_encoder(out_channels + concat_channels, out_channels, norm)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, encoder_out = x
        up1 = F.interpolate(
            encoder_out,
            scale_factor=self.stride[0],
            mode="trilinear",
            align_corners=True,
        )
        x1 = self.conv1(up1)
        x1 = torch.cat((x1, features), dim=1)
        x1 = self.conv2(x1)
        return x1


class reg_decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        stride: Sequence[int] = (2, 2, 2),
        kernel: Sequence[int] = (3, 3, 3),
    ) -> None:
        super(reg_decoder, self).__init__()
        self.conv1 = reg_encoder(in_channels, out_channels, norm, kernel=kernel)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, encoder_out = x
        up1 = F.interpolate(
            encoder_out,
            scale_factor=self.stride[0],
            mode="trilinear",
            align_corners=True,
        )
        x1 = self.conv1(up1)
        x1 = torch.cat((x1, features), dim=1)
        return x1


class BaseSeg(nn.Module):
    def __init__(self, cfg):
        super(BaseSeg, self).__init__()
        self.segnet = monai.networks.nets.UNet(
            spatial_dims=cfg.model.baseseg.spatial_dim,  # spatial dims
            in_channels=cfg.model.baseseg.in_channel,  # input channels
            out_channels=cfg.model.baseseg.out_channel,  # output channels
            channels=cfg.model.baseseg.channels,  # channel sequence
            strides=cfg.model.baseseg.strides,  # convolutional strides
            dropout=cfg.model.baseseg.dropout,
            act=cfg.model.baseseg.act,
            norm=cfg.model.baseseg.norm,
        )

    def forward(self, x):
        return self.segnet(x)


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), (
            "At least one weight factor should be != 0.0"
        )
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), (
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        )
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1,) * len(args[0])
        else:
            weights = self.weight_factors

        # for i, inputs in enumerate(zip(*args)):
        #     if weights[i] != 0.0:
        #         print(self.loss, weights[i], self.loss(*inputs))

        return sum(
            [
                weights[i] * self.loss(*inputs)
                for i, inputs in enumerate(zip(*args))
                if weights[i] != 0.0
            ]
        )
