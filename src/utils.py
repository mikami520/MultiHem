#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 18:33:19
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-03-08 23:48:06
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
from os.path import join
import glob
import shutil
import logging
from typing import Union, Sequence

plt.switch_backend("agg")

__all__ = [
    "check_num_workers",
    "plot_progress",
    "setup_logger",
    "make_if_dont_exist",
    "create_batch_generator",
    "compute_train_intensity_stats",
    "DataHandler",
]

seg_availabilities = ["00", "01", "10", "11"]


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


def create_batch_generator(dataloader_subdivided, seg=False):
    """
    Create a batch generator that samples data pairs with various segmentation availabilities.

    Arguments:
        dataloader_subdivided : a mapping from the labels in seg_availabilities to dataloaders
        weights : a list of probabilities, one for each label in seg_availabilities;
                  if not provided then we weight by the number of data items of each type,
                  effectively sampling uniformly over the union of the datasets

    Returns: batch_generator
        A function that accepts a number of batches to sample and that returns a generator.
        The generator will weighted-randomly pick one of the seg_availabilities and
        yield the next batch from the corresponding dataloader.
    """
    if not seg:
        weights = np.array([len(dataloader_subdivided[s]) for s in seg_availabilities])
    else:
        weights = [0] + [len(dataloader_subdivided[s]) for s in seg_availabilities[1:]]

    weights = np.array(weights)
    weights = weights / weights.sum()
    dataloader_subdivided_as_iterators = {
        s: iter(d) for s, d in dataloader_subdivided.items()
    }

    def batch_generator(num_batches_to_sample):
        for _ in range(num_batches_to_sample):
            seg_availability = np.random.choice(seg_availabilities, p=weights)
            try:
                yield next(dataloader_subdivided_as_iterators[seg_availability])
            except StopIteration:  # If dataloader runs out, restart it
                dataloader_subdivided_as_iterators[seg_availability] = iter(
                    dataloader_subdivided[seg_availability]
                )
                yield next(dataloader_subdivided_as_iterators[seg_availability])

    return batch_generator


def compute_train_intensity_stats(train_data_list):
    intensities = []
    for data in train_data_list:
        image = np.load(data["image"])  # Load training images
        intensities.append(image.flatten())  # Collect all pixel values
    intensities = np.concatenate(intensities)
    mean, std = np.mean(intensities), np.std(intensities)
    return mean, std


class DataHandler:
    def __init__(self, cfg):
        self.path_img = cfg.data.img_dir
        self.path_seg = cfg.data.seg_dir
        self.n_samples = cfg.data.n_samples
        self.postfix = cfg.data.postfix

    def _subdivide_data(self, paired_data):
        data = {"00": [], "01": [], "10": [], "11": []}
        for ele in paired_data:
            if "seg1" in ele.keys() and "seg2" in ele.keys():
                data["11"].append(ele)
            elif "seg1" in ele.keys():
                data["10"].append(ele)
            elif "seg2" in ele.keys():
                data["01"].append(ele)
            else:
                data["00"].append(ele)
        return data

    def _pair_data(self, raw_data):
        data = []
        for ele1 in raw_data:
            id1 = ele1["id"]
            img1 = ele1["img"]
            seg1 = ele1.get("seg", None)
            subtype = ele1["subtype"]
            for ele2 in raw_data:
                ele = {}
                id2 = ele2["id"]
                if id1 == id2:
                    continue

                ele["img1"] = img1
                ele["img2"] = ele2["img"]
                ele["subtype"] = subtype
                seg2 = ele2.get("seg", None)

                if seg1 is not None:
                    ele["seg1"] = seg1

                if seg2 is not None:
                    ele["seg2"] = seg2

                data.append(ele)

        return data

    def _get_paths(self):
        data = []
        all_imgs = glob.glob(join(self.path_img, f"*.{self.postfix}"))
        if len(all_imgs) == 0:
            raise ValueError("No images found in the path")

        for i in all_imgs:
            if len(data) >= self.n_samples:
                return data
            ele = {}
            ele["img"] = i
            filename = i.split("/")[-1]
            seg_path = join(self.path_seg, filename)
            subtype = filename.split(".")[0].split("_")[-1]
            ele["id"] = filename.split(".")[0]
            ele["subtype"] = int(subtype)
            if os.path.exists(seg_path):
                ele["seg"] = seg_path

            data.append(ele)

        return data
