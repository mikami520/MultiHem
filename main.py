#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:34:17
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-07 22:08:02
FilePath     : /Downloads/MultiHem/main.py
Description  : Main Function of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from src import Trainer, BaseTrainer
import argparse
import os
from omegaconf import OmegaConf


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None, type=str, help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="use this if you want to continue a training",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device to use for training (default: cuda)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="use this if you want to test the model",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_command()
    cfg_path = args.cfg
    device = args.device
    check_test = args.test
    resume = args.resume
    if cfg_path is not None:
        if os.path.exists(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        else:
            raise FileNotFoundError(f"config file {cfg_path} not found")
    else:
        raise ValueError("config file not specified")

    trainer = Trainer(cfg, device, check_test)

    if check_test:
        trainer.test()
    else:
        if resume:
            trainer.resume()
            resume = False
        else:
            trainer.prepare_dir()
            trainer.prepare_logger()
            trainer.train()


if __name__ == "__main__":
    main()
