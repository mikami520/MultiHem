#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 21:08:57
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-19 21:13:25
FilePath     : /MultiHem/src/trainer.py
Description  : Trainer of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from src.model import MultiHem
from src.utils import setup_logger, plot_progress, make_if_dont_exist


class Trainer:
    def __init__(self, *args, **kwargs):
        self.model = MultiHem()
        self.optimizer = None
        self.scheduler = None
        self.device = None

    def train(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
