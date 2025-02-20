#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:33:19
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-20 03:09:10
FilePath     : /MultiHem/src/utils.py
Description  : Help Functions of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
import logging
from typing import Union, Sequence

plt.switch_backend("agg")

__all__ = [
    "check_num_workers",
    "plot_progress",
    "setup_logger",
    "make_if_dont_exist",
]


def check_num_workers(cfg, train_reader):
    min_span = np.inf
    best_num_workers = 0
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(
            train_reader,
            shuffle=True,
            num_workers=num_workers,
            batch_size=cfg.train.batch_size,
            pin_memory=True,
        )
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        if (end - start) < min_span:
            min_span = end - start
            best_num_workers = num_workers

    return best_num_workers


def plot_progress(
    logger: logging.Logger,
    save_dir: str,
    train_loss: Sequence[Sequence[Union[int, float]]],
    val_loss: Sequence[Sequence[Union[int, float]]],
    name: str,
) -> None:
    """
    Should probably by improved
    :return:
    """
    assert len(train_loss) != 0
    train_loss = np.array(train_loss)
    try:
        font = {"weight": "normal", "size": 18}

        matplotlib.rc("font", **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax.plot(train_loss[:, 0], train_loss[:, 1], color="b", ls="-", label="loss_tr")
        if len(val_loss) != 0:
            val_loss = np.array(val_loss)
            ax.plot(val_loss[:, 0], val_loss[:, 1], color="r", ls="-", label="loss_val")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)
    except ImportError as e:
        logger.info(f"ImportError: failed to plot {name} training progress: {e}")
        raise e
    except Exception as e:
        logger.info(f"Failed to plot {name} training progress: {e}")
        raise e


def setup_logger(
    logger_name: str, log_file: str, level: int = logging.INFO
) -> logging.Logger:
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_setup.setLevel(level)
    log_setup.propagate = False
    if not log_setup.handlers:
        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        log_setup.addHandler(fileHandler)
        log_setup.addHandler(streamHandler)

    return log_setup


def make_if_dont_exist(folder_path: str, overwrite: bool = False):
    if os.path.exists(folder_path):
        if not overwrite:
            print(f"{folder_path} exists, no overwrite here.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")
