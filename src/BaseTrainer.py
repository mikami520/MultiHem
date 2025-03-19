#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-03-08 22:04:58
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-03-19 17:19:58
FilePath     : /Downloads/MultiHem/src/BaseTrainer.py
Description  : Trainer script for base model
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import os
import datetime
import numpy as np
from tqdm.auto import tqdm
import torch
from src.utils import make_if_dont_exist, setup_logger, DataHandler, plot_progress
from src.model import BaseSeg
from src.loss import DiceLoss
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Spacingd,
    ResizeD,
    SpatialPadD,
    Compose,
)
from monai.data import CacheDataset, DataLoader
from monai.data.utils import partition_dataset

TRANSFORMS = Compose(
    [
        LoadImaged(keys=["img", "seg"], image_only=False),
        EnsureChannelFirstd(keys=["img", "seg"]),
        # Resample to 0.5mm isotropic spacing
        Spacingd(
            keys=["img", "seg"],
            pixdim=(0.5, 0.5, 5.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["img"], a_min=-100, a_max=300, b_min=0, b_max=1, clip=True
        ),
        # Resize XY but keep Z unchanged
        ResizeD(
            keys=["img", "seg"],
            spatial_size=(256, 256, -1),
            mode=("trilinear", "nearest"),
        ),
        # Pad Z-dimension to 128 slices
        SpatialPadD(keys=["img", "seg"], spatial_size=(256, 256, 128)),
    ]
)


class BaseTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.train_dir = "exp"
        self.num_classes = self.cfg.model.segnet.out_channel
        self.exp_dir = os.path.join(self.train_dir, self.cfg.exp_name)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.plot_dir = os.path.join(self.exp_dir, "plot")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        self.last_epoch_train = 0
        self.prepare_network()
        self.prepare_optimizer()
        self.prepare_loss()
        self.prepare_dataloader()

    def prepare_dir(self):
        make_if_dont_exist(self.train_dir)
        make_if_dont_exist(self.exp_dir)
        make_if_dont_exist(self.log_dir)
        make_if_dont_exist(self.model_dir)
        make_if_dont_exist(self.plot_dir)
        make_if_dont_exist(self.checkpoint_dir)

    def prepare_network(self):
        self.net = BaseSeg(self.cfg).to(self.device)
        self.net.eval()

    def prepare_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.cfg.model.baseseg.lr
        )

    def prepare_loss(self):
        self.criteria = DiceLoss()
        self.train_losses = []
        self.valid_losses = []
        self.best_loss = np.inf

    def prepare_logger(self, resume: bool = False):
        datetime_object = (
            "training_log_"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            + ".log"
        )

        self.logger = setup_logger(
            "BaselineSeg", os.path.join(self.log_dir, datetime_object)
        )
        if not resume:
            self.logger.info(f"\tStart {self.cfg.experiment} Training From Scratch")
        else:
            self.logger.info(
                f"\tResuming Training from Epoch {self.last_epoch_train + 1}"
            )

    def prepare_dataloader(self):
        handler = DataHandler(self.cfg)
        data_path = handler._get_paths()

        data_train, data_valid = partition_dataset(data_path, ratios=(8, 2))
        self.dataloader_train = DataLoader(
            CacheDataset(
                data=data_train,
                transform=TRANSFORMS,
                cache_num=16,
            ),
            batch_size=int(self.cfg.model.baseseg.batch_size),
            num_workers=0,
            shuffle=True,
            pin_memory=True,
        )
        self.dataloader_valid = DataLoader(
            CacheDataset(
                data=data_valid,
                transform=TRANSFORMS,
                cache_num=16,
            ),
            batch_size=int(self.cfg.model.baseseg.batch_size) * 2,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def train(self):
        self.logger.info("-----" * 10)
        self.logger.info("\tYou are working on the baseline")
        self.logger.info("-----" * 10)
        for epoch in range(self.last_epoch_train, self.cfg.train.epochs):
            self.logger.info(f"\tEpoch {epoch + 1}/{self.cfg.train.epochs}")
            self.last_epoch_train = epoch
            losses_tr = []
            self.logger.info("-----" * 10)
            with tqdm(
                self.dataloader_train,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.net.train()
                    self.optimizer.zero_grad()
                    img, seg = data["img"].to(self.device), data["seg"].to(self.device)
                    pred = self.net(img)
                    loss = self.criteria(pred, seg)
                    loss.backward()
                    self.optimizer.step()
                    losses_tr.append(loss.item())
                    tdata.set_postfix(loss=loss.item())

            avg_loss = np.mean(losses_tr, axis=0)
            self.train_losses.append([epoch + 1, avg_loss])
            self.logger.info(f"\tTraining Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.net.eval()
                losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_valid,
                        unit="batch",
                        desc="Validation",
                    ) as vdata:
                        for data in vdata:
                            img, seg = (
                                data["img"].to(self.device),
                                data["seg"].to(self.device),
                            )
                            pred = self.net(img)
                            loss = self.criteria(pred, seg)
                            losses_val.append(loss.item())
                            vdata.set_postfix(loss=loss.item())

                avg_loss = np.mean(losses_val, axis=0)
                self.valid_losses.append([epoch + 1, avg_loss])
                self.logger.info(f"\tValidation Loss: {avg_loss:.4f}")
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_loss:.4f}"
                    )
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(self.model_dir, "best_baseline.pt"),
                    )

            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses,
                self.valid_losses,
                "Baseline",
            )
            del pred, loss, losses_tr, losses_val, avg_loss
            torch.cuda.empty_cache()
            self.logger.info(f"\tSaving Checkpoint for Epoch {epoch + 1}")
            self.save_checkpoint()

    def save_checkpoint(self):
        torch.save(
            {
                "epoch": self.last_epoch_train,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "train_losses": self.train_losses,
                "valid_losses": self.valid_losses,
            },
            os.path.join(self.checkpoint_dir, "checkpoint.pt"),
        )

    def load_checkpoint(self):
        checkpoint = (
            torch.load(
                os.path.join(self.checkpoint_dir, "checkpoint.pt"),
                map_location=self.device,
                weights_only=False,
            )
            if os.path.exists(os.path.join(self.checkpoint_dir, "checkpoint.pt"))
            else None
        )
        if checkpoint is None:
            print("No checkpoint found, training from scratch")
            return

        # epoch
        self.last_epoch_train = checkpoint["epoch"] + 1

        # model
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint["best_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.valid_losses = checkpoint["valid_losses"]

    def resume(self):
        self.load_checkpoint()
        self.prepare_logger(resume=True)
        self.train()
