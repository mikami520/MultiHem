#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:29:30
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-03-19 17:16:03
FilePath     : /Downloads/MultiHem/src/model.py
Description  : Backbone Architectures of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

__all__ = [
    "same_padding",
    "SegNet",
    "RegNet",
    "Fusion",
    "Classifier",
    "BaseSeg",
    "seg_encoder",
    "reg_encoder",
    "seg_decoder",
    "seg_decoder_last",
    "reg_decoder",
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


class Fusion(nn.Module):
    def __init__(
        self,
        seg_encode_layers: Sequence[int],
        reg_encode_layers: Sequence[int],
        stride: Sequence[int],
    ) -> None:
        super(Fusion, self).__init__()
        assert len(seg_encode_layers) == len(reg_encode_layers)
        self.stride = copy.deepcopy(stride)
        self.stride.insert(0, 1)
        att_list = []
        for l_seg, l_reg in zip(seg_encode_layers, reg_encode_layers):
            att_list.append(
                nn.Sequential(
                    nn.Conv3d(l_seg + l_reg, 128, kernel_size=(1, 1, 1)),
                    ChannelAttention(128),
                    SpatialAttention(),
                    nn.Conv3d(128, 128, kernel_size=(1, 1, 1)),
                )
            )
        self.att_list = nn.ModuleList(att_list)
        self.smooth_conv = nn.Conv3d(128 * len(att_list), 128, kernel_size=(1, 1, 1))

    def forward(
        self, x_seg: Sequence[torch.Tensor], x_reg: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        assert len(x_seg) == len(x_reg)
        x_tmp = []
        for i, att in enumerate(self.att_list):
            assert x_seg[i].shape[2:] == x_reg[i].shape[2:], (
                f"Shape mismatch between x_seg[{i}] and x_reg[{i}]: "
                f"{x_seg[i].shape[2:]} != {x_reg[i].shape[2:]}"
            )
            x_tmp.append(
                att(torch.cat((x_seg[i], x_reg[i]), dim=1))
            )  # [B, 128, Hi, Wi, Di]

        x_out = []
        for i, x in enumerate(x_tmp):
            x_out.append(
                F.interpolate(
                    x,
                    scale_factor=int(np.cumprod(self.stride)[i]),
                    mode="trilinear",
                    align_corners=True,
                )
            )  # [B, 128, H0, W0, D0]

        x_out = torch.cat(x_out, dim=1)  # [B, 128 * len(x_tmp), H0, W0, D0]
        x_out = self.smooth_conv(x_out)
        return x_out  # [B, 128, H0, W0, D0]


class ChannelAttention(nn.Module):
    def __init__(self, in_channel: int, reduction: int = 16) -> None:
        super(ChannelAttention, self).__init__()
        self.ffn = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // reduction, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(in_channel // reduction, in_channel, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.ffn(F.adaptive_avg_pool3d(x, (1, 1, 1)))  # [B, C, 1, 1, 1]
        max_out = self.ffn(F.adaptive_max_pool3d(x, (1, 1, 1)))  # [B, C, 1, 1, 1]
        att_map = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1, 1]
        return x * att_map


class SpatialAttention(nn.Module):
    def __init__(
        self,
        kernel: Sequence[int] | int = 3,
        dilation: Sequence[int] | int = 1,
    ) -> None:
        super(SpatialAttention, self).__init__()
        padding = same_padding(kernel, dilation)
        self.conv = nn.Conv3d(
            2, 1, kernel_size=kernel, padding=padding, dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W, D]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W, D]
        feat_cat = torch.cat((avg_out, max_out), dim=1)  # [B, 2, H, W, D]
        att_map = self.sigmoid(self.conv(feat_cat))  # [B, 1, H, W, D]
        return x * att_map


class Classifier(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_classes: int,
        hidden_dim: int = 256,  # Lower hidden dimension for efficiency.
        num_transformer_layers: int = 2,  # Using a single layer.
        num_heads: int = 4,  # Fewer attention heads.
        dropout: float = 0.1,
    ):
        super(Classifier, self).__init__()
        # Project fused features to a lower-dimensional space.
        self.conv_proj = nn.Conv3d(in_channel, hidden_dim, kernel_size=1)

        # Learnable classification token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Positional embeddings will be dynamically initialized based on token count.
        self.pos_embed = None
        self.pos_embed_initialized = False

        # Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Final classification head.
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channel, H, W, D] from the fusion module.
        B, C, H, W, D = x.shape
        x = self.conv_proj(x)  # [B, hidden_dim, H, W, D]

        # Flatten spatial dimensions to create a sequence of tokens.
        x = x.flatten(2)  # [B, hidden_dim, N] where N = H * W * D.
        x = x.transpose(1, 2)  # [B, N, hidden_dim]

        # Initialize positional embeddings if not done already or if the sequence length changes.
        if not self.pos_embed_initialized or self.pos_embed.shape[1] != (
            x.shape[1] + 1
        ):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, x.shape[1] + 1, x.shape[2], device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_embed_initialized = True

        # Expand the classification token for the batch.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]

        # Concatenate the classification token with the flattened tokens.
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, hidden_dim]

        # Add positional embeddings.
        x = x + self.pos_embed
        x = self.dropout(x)

        # Prepare data for transformer: [sequence length, batch size, hidden_dim]
        x = x.transpose(0, 1)  # [1+N, B, hidden_dim]

        # Process through the transformer encoder.
        x = self.transformer(x)  # [1+N, B, hidden_dim]

        # Use the classification token for the final prediction.
        cls_out = x[0]  # [B, hidden_dim]
        logits = self.classifier(cls_out)  # [B, num_classes]
        return logits


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
