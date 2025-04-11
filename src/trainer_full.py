#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 21:08:57
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-11 14:23:33
FilePath     : /Downloads/MultiHem/src/trainer_full.py
Description  : Trainer of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import os
import datetime
import numpy as np
from tqdm.auto import tqdm
from src.utils import (
    setup_logger,
    plot_progress,
    make_if_dont_exist,
    DataHandler,
    create_batch_generator,
)
from src.model import (
    SegNet,
    RegNet,
    Classifier,
    DeepSupervisionWrapper,
)
from src.loss import SegLoss, RegLoss, ClassLoss, DiceLoss
import monai
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Spacingd,
    ResizeD,
    SpatialPadD,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandBiasFieldd,
    RandShiftIntensityd,
    ConcatItemsD,
    DeleteItemsD,
    LambdaD,
    Compose,
)
from monai.data import CacheDataset, DataLoader
from monai.data.utils import partition_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceHelper, HausdorffDistanceMetric
import json
import ants

TRANSFORMS = {
    "train": Compose(
        transforms=[
            LoadImaged(
                keys=["img1", "seg1", "img2", "seg2"],
                image_only=True,
                allow_missing_keys=True,
            ),
            EnsureChannelFirstd(
                keys=["img1", "seg1", "img2", "seg2"], allow_missing_keys=True
            ),
            # Resample to 0.5mm isotropic spacing
            Spacingd(
                keys=["img1", "seg1", "img2", "seg2"],
                pixdim=(0.5, 0.5, 5.0),
                mode=("bilinear", "nearest", "bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            # ✅ Apply intensity augmentation in HU space
            RandShiftIntensityd(
                keys=["img1", "img2"],
                offsets=50,
                prob=0.3,
            ),  # Since HU is in [-1000, 3000]
            RandGaussianNoised(
                keys=["img1", "img2"],
                std=20.0,
                prob=0.2,
            ),
            RandBiasFieldd(keys=["img1", "img2"], coeff_range=(0.0, 0.1), prob=0.2),
            NormalizeIntensityd(
                keys=["img1", "img2"],
                nonzero=True,  # ✅ Use only nonzero voxels
                channel_wise=False,  # Per-image, not per-channel
            ),
            LambdaD(
                keys=["img1", "img2"],
                func=lambda x: (x - x.min())
                / (x.max() - x.min() + 1e-8),  # normalizes to [0, 1]
            ),
            # Resize XY but keep Z unchanged
            ResizeD(
                keys=["img1", "seg1", "img2", "seg2"],
                spatial_size=(128, 128, -1),
                mode=("trilinear", "nearest", "trilinear", "nearest"),
                allow_missing_keys=True,
            ),
            # Pad Z-dimension to 128 slices
            SpatialPadD(
                keys=["img1", "seg1", "img2", "seg2"],
                spatial_size=(128, 128, 128),
                allow_missing_keys=True,
            ),
            RandRotated(
                keys=["img1", "seg1", "img2", "seg2"],
                range_z=np.pi / 12,
                mode=["bilinear", "nearest", "bilinear", "nearest"],
                prob=0.5,
                keep_size=True,
                allow_missing_keys=True,
            ),
            RandZoomd(
                keys=["img1", "seg1", "img2", "seg2"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
                mode=["trilinear", "nearest", "trilinear", "nearest"],
                keep_size=True,
                allow_missing_keys=True,
            ),
            ConcatItemsD(keys=["img1", "img2"], name="img12", dim=0),
            DeleteItemsD(keys=["img1", "img2"]),
        ]
    ),
    "valid": Compose(
        transforms=[
            LoadImaged(
                keys=["img1", "seg1", "img2", "seg2"],
                image_only=True,
                allow_missing_keys=True,
            ),
            EnsureChannelFirstd(
                keys=["img1", "seg1", "img2", "seg2"],
                allow_missing_keys=True,
            ),
            # Resample to 0.5mm isotropic spacing
            Spacingd(
                keys=["img1", "seg1", "img2", "seg2"],
                pixdim=(0.5, 0.5, 5.0),
                mode=("bilinear", "nearest", "bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(
                keys=["img1", "img2"],
                nonzero=True,  # ✅ Use only nonzero voxels
                channel_wise=False,  # Per-image, not per-channel
            ),
            LambdaD(
                keys=["img1", "img2"],
                func=lambda x: (x - x.min())
                / (x.max() - x.min() + 1e-8),  # normalizes to [0, 1]
            ),
            # Resize XY but keep Z unchanged
            ResizeD(
                keys=["img1", "seg1", "img2", "seg2"],
                spatial_size=(128, 128, -1),
                mode=("trilinear", "nearest", "trilinear", "nearest"),
                allow_missing_keys=True,
            ),
            # Pad Z-dimension to 128 slices
            SpatialPadD(
                keys=["img1", "seg1", "img2", "seg2"],
                spatial_size=(128, 128, 128),
                allow_missing_keys=True,
            ),
            ConcatItemsD(keys=["img1", "img2"], name="img12", dim=0),
            DeleteItemsD(keys=["img1", "img2"]),
        ]
    ),
    "valid_seg": Compose(
        transforms=[
            LoadImaged(
                keys=["img", "seg"],
                image_only=False,
            ),
            EnsureChannelFirstd(
                keys=["img", "seg"],
            ),
            # Resample to 0.5mm isotropic spacing
            Spacingd(
                keys=["img", "seg"],
                pixdim=(0.5, 0.5, 5.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(
                keys=["img"],
                nonzero=True,  # ✅ Use only nonzero voxels
                channel_wise=False,  # Per-image, not per-channel
            ),
            # Resize XY but keep Z unchanged
            ResizeD(
                keys=["img", "seg"],
                spatial_size=(128, 128, -1),
                mode=("trilinear", "nearest"),
            ),
            # Pad Z-dimension to 128 slices
            SpatialPadD(
                keys=["img", "seg"],
                spatial_size=(128, 128, 128),
            ),
        ]
    ),
}


class FullTrainer:
    def __init__(self, cfg, device, test=False):
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
        self.prepare_dataloader(test=test)

    def prepare_dir(self):
        make_if_dont_exist(self.train_dir)
        make_if_dont_exist(self.exp_dir)
        make_if_dont_exist(self.log_dir)
        make_if_dont_exist(self.model_dir)
        make_if_dont_exist(self.plot_dir)
        make_if_dont_exist(self.checkpoint_dir)

    def prepare_network(self):
        # Initialize the segmentation network
        self.segnet = SegNet(
            in_channel=self.cfg.model.segnet.in_channel,
            out_channel=self.cfg.model.segnet.out_channel,
            encode_layers=self.cfg.model.segnet.encode_layers,
            decode_layers=self.cfg.model.segnet.decode_layers,
            stride=self.cfg.model.segnet.stride,
            dropout=self.cfg.model.segnet.dropout,
            norm=self.cfg.model.segnet.norm,
        ).to(self.device)

        self.regnet = RegNet(
            in_channel=self.cfg.model.regnet.in_channel,
            out_channel=self.cfg.model.regnet.out_channel,
            encode_layers=self.cfg.model.regnet.encode_layers,
            decode_layers=self.cfg.model.regnet.decode_layers,
            stride=self.cfg.model.regnet.stride,
            dropout=self.cfg.model.regnet.dropout,
            norm=self.cfg.model.regnet.norm,
        ).to(self.device)

        self.classnet = Classifier(
            in_channels=self.cfg.model.classnet.in_channel,
            num_classes=self.cfg.model.classnet.out_channel,
            res_channels=self.cfg.model.classnet.res_channel,
            dropout_p=self.cfg.model.classnet.dropout,
            deep_supervision=False,
        ).to(self.device)

        self.segnet.eval()
        self.regnet.eval()
        self.classnet.eval()

    def prepare_optimizer(self):
        self.seg_optim = torch.optim.Adam(
            self.segnet.parameters(),
            lr=self.cfg.model.segnet.lr,
            weight_decay=self.cfg.model.segnet.weight_decay,
        )
        self.reg_optim = torch.optim.Adam(
            self.regnet.parameters(),
            lr=self.cfg.model.regnet.lr,
            weight_decay=self.cfg.model.regnet.weight_decay,
        )
        self.class_optim = torch.optim.Adam(
            self.classnet.parameters(),
            lr=self.cfg.model.classnet.lr,
            weight_decay=self.cfg.model.classnet.weight_decay,
        )
        self.seg_lr_scheduler = CosineAnnealingLR(
            self.seg_optim, T_max=self.cfg.train.epochs, eta_min=1e-6
        )
        self.reg_lr_scheduler = CosineAnnealingLR(
            self.reg_optim, T_max=self.cfg.train.epochs, eta_min=1e-6
        )
        self.class_lr_scheduler = CosineAnnealingLR(
            self.class_optim, T_max=self.cfg.train.epochs, eta_min=1e-6
        )

    def prepare_loss(self):
        w_super, _ = self.cfg.model.segnet.loss_weights
        w_sim, w_penal, w_ana = self.cfg.model.regnet.loss_weights
        self.seg_criterion = SegLoss(w_super, w_ana)
        self.reg_criterion = RegLoss(
            w_sim, w_penal, w_ana, num_classes=self.num_classes
        )
        self.class_criterion = ClassLoss(label_smoothing=0.1)
        # self.class_criterion = DeepSupervisionWrapper(
        #     loss_class,
        #     weight_factors=weights_class,
        # )
        self.dice_loss = DiceLoss()
        self.train_losses_reg = []
        self.valid_losses_reg = []
        self.train_losses_seg = []
        self.valid_losses_seg = []
        self.train_losses_class = []
        self.valid_losses_class = []
        self.best_reg_loss = np.inf
        self.best_seg_loss = np.inf
        self.best_class_loss = np.inf

    def prepare_logger(self, resume: bool = False):
        datetime_object = (
            "training_log_"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            + ".log"
        )

        self.logger = setup_logger(
            "DeepControlV2", os.path.join(self.log_dir, datetime_object)
        )
        if not resume:
            self.logger.info(f"\tStart {self.cfg.experiment} Training From Scratch")
        else:
            self.logger.info(
                f"\tResuming Training from Epoch {self.last_epoch_train + 1}"
            )

    def prepare_dataloader(self, test=False):
        handler = DataHandler(self.cfg)
        data_path = handler._get_paths()
        if not test:
            data_seg_avail = list(filter(lambda x: "seg" in x.keys(), data_path))
            data_seg_not_avail = list(
                filter(lambda x: "seg" not in x.keys(), data_path)
            )

            data_seg_avail_tr, data_seg_avail_val = partition_dataset(
                data_seg_avail, ratios=(8, 2)
            )

            self.dataloader_seg_val = DataLoader(
                CacheDataset(
                    data=data_seg_avail_val,
                    transform=TRANSFORMS["valid_seg"],
                    cache_num=16,
                ),
                batch_size=self.cfg.model.segnet.batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
            )

            data_both = data_seg_avail_tr + data_seg_not_avail
            data_both_tr, data_both_val = monai.data.utils.partition_dataset(
                data_both, ratios=(8, 2), shuffle=False
            )

            paired_data_both_tr = handler._pair_data(data_both_tr)
            paired_data_both_val = handler._pair_data(data_both_val)

            subdivided_data_both_tr = handler._subdivide_data(paired_data_both_tr)
            subdivided_data_both_val = handler._subdivide_data(paired_data_both_val)

            subdivided_dataset_both_tr = {
                seg_availability: monai.data.CacheDataset(
                    data=data_list, transform=TRANSFORMS["train"], cache_num=16
                )
                for seg_availability, data_list in subdivided_data_both_tr.items()
            }
            subdivided_dataset_both_val = {
                seg_availability: monai.data.CacheDataset(
                    data=data_list, transform=TRANSFORMS["valid"], cache_num=16
                )
                for seg_availability, data_list in subdivided_data_both_val.items()
            }

            dataloader_reg_tr = {
                seg_availability: (
                    monai.data.DataLoader(
                        dataset,
                        batch_size=int(self.cfg.model.regnet.batch_size),
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True,
                    )
                    if len(dataset) > 0
                    else []
                )  # empty dataloaders are not a thing-- put an empty list if needed
                for seg_availability, dataset in subdivided_dataset_both_tr.items()
            }

            dataloader_reg_val = {
                seg_availability: (
                    monai.data.DataLoader(
                        dataset,
                        batch_size=self.cfg.model.regnet.batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=True,
                    )
                    if len(dataset) > 0
                    else []
                )  # empty dataloaders are not a thing-- put an empty list if needed
                for seg_availability, dataset in subdivided_dataset_both_val.items()
            }

            self.batch_gen_reg_tr = create_batch_generator(dataloader_reg_tr)
            self.batch_gen_reg_val = create_batch_generator(dataloader_reg_val)
            self.batch_gen_seg_tr = create_batch_generator(dataloader_reg_tr, seg=True)
        else:
            self.dataloader_test = CacheDataset(
                data=data_path,
                transform=TRANSFORMS["valid_seg"],
                cache_num=16,
            )

    def swap_training(self, nets_train, nets_freeze):
        for net in nets_train:
            for param in net.parameters():
                param.requires_grad = True
            net.train()
        for net in nets_freeze:
            for param in net.parameters():
                param.requires_grad = False
            net.eval()

    def train(self):
        for epoch in range(self.last_epoch_train, self.cfg.train.epochs):
            self.logger.info("-----" * 10)
            self.logger.info(f"\tEpoch {epoch + 1}/{self.cfg.train.epochs}")
            self.last_epoch_train = epoch
            reg_losses_tr = []
            self.swap_training(
                nets_train=[self.regnet],
                nets_freeze=[self.segnet, self.classnet],
            )
            with tqdm(
                self.batch_gen_reg_tr(self.cfg.model.regnet.n_batch_per_epoch),
                total=self.cfg.model.regnet.n_batch_per_epoch,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.reg_optim.zero_grad()
                    img_pair = data["img12"].to(self.device).float()

                    disp, _ = self.regnet(img_pair)
                    if "seg1" in data.keys():
                        seg1 = data["seg1"]
                        seg1 = seg1.to(self.device)
                        seg1 = monai.networks.one_hot(
                            seg1, num_classes=self.num_classes
                        )
                    else:
                        seg1, _ = self.segnet(img_pair[:, [0], :, :, :])
                        seg1 = torch.softmax(seg1, dim=1)

                    if "seg2" in data.keys():
                        seg2 = data["seg2"]
                        seg2 = seg2.to(self.device)
                        seg2 = monai.networks.one_hot(
                            seg2, num_classes=self.num_classes
                        )
                    else:
                        seg2, _ = self.segnet(img_pair[:, [1], :, :, :])
                        seg2 = torch.softmax(seg2, dim=1)

                    reg_loss = self.reg_criterion(img_pair, disp, seg1, seg2)
                    reg_loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.regnet.parameters(), 12)
                    # torch.nn.utils.clip_grad_norm_(self.classnet.parameters(), 12)
                    self.reg_optim.step()
                    reg_losses_tr.append(reg_loss.item())
                    tdata.set_postfix(
                        {
                            "reg_loss": reg_loss.item(),
                        }
                    )
            avg_reg_loss = np.mean(reg_losses_tr, axis=0)
            self.train_losses_reg.append([epoch + 1, avg_reg_loss])
            self.logger.info(f"\tReg Train Loss: {avg_reg_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.regnet.eval()
                reg_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.batch_gen_reg_val(self.cfg.model.regnet.n_batch_per_epoch),
                        total=self.cfg.model.regnet.n_batch_per_epoch,
                        unit="batch",
                        desc="Validation",
                    ) as tdata:
                        for data in tdata:
                            img_pair = data["img12"].to(self.device).float()
                            disp, _ = self.regnet(img_pair)

                            if "seg1" in data.keys():
                                seg1 = data["seg1"]
                                seg1 = seg1.to(self.device)
                                seg1 = monai.networks.one_hot(
                                    seg1, num_classes=self.num_classes
                                )
                            else:
                                seg1, _ = self.segnet(img_pair[:, [0], :, :, :])
                                seg1 = torch.softmax(seg1, dim=1)

                            if "seg2" in data.keys():
                                seg2 = data["seg2"]
                                seg2 = seg2.to(self.device)
                                seg2 = monai.networks.one_hot(
                                    seg2, num_classes=self.num_classes
                                )
                            else:
                                seg2, _ = self.segnet(img_pair[:, [1], :, :, :])
                                seg2 = torch.softmax(seg2, dim=1)

                            reg_loss = self.reg_criterion(img_pair, disp, seg1, seg2)
                            reg_losses_val.append(reg_loss.item())
                            tdata.set_postfix({"reg_loss": reg_loss.item()})
                avg_reg_loss = np.mean(reg_losses_val, axis=0)
                self.valid_losses_reg.append([epoch + 1, avg_reg_loss])
                self.logger.info(f"\tReg Valid Loss: {avg_reg_loss:.4f}")
                if avg_reg_loss < self.best_reg_loss:
                    self.best_reg_loss = avg_reg_loss
                    self.logger.info(
                        f"\tSaving Best Reg Model with Loss: {self.best_reg_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_seg": self.segnet.state_dict(),
                            "weights_reg": self.regnet.state_dict(),
                        },
                        os.path.join(self.model_dir, "best.pt"),
                    )

            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_reg,
                self.valid_losses_reg,
                "Registration_Loss",
            )
            self.reg_lr_scheduler.step()
            del (
                img_pair,
                disp,
                seg1,
                seg2,
                reg_loss,
                reg_losses_tr,
                reg_losses_val,
            )
            torch.cuda.empty_cache()
            self.logger.info("-----" * 10)
            # Train Segmentation Network
            self.swap_training(
                nets_train=[self.segnet],
                nets_freeze=[self.regnet, self.classnet],
            )
            # ---------------------------------------------------------
            #     seg_net training, with reg_net and class_net frozen
            # ---------------------------------------------------------
            seg_losses_tr = []
            with tqdm(
                self.batch_gen_seg_tr(self.cfg.model.segnet.n_batch_per_epoch),
                total=self.cfg.model.segnet.n_batch_per_epoch,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.seg_optim.zero_grad()
                    img_pair = data["img12"].to(self.device).float()
                    disp, _ = self.regnet(img_pair)
                    pred_seg1, _ = self.segnet(img_pair[:, [0], :, :, :])
                    pred_seg2, _ = self.segnet(img_pair[:, [1], :, :, :])
                    pred_seg1 = torch.softmax(pred_seg1, dim=1)
                    pred_seg2 = torch.softmax(pred_seg2, dim=1)
                    if "seg1" in data.keys() and "seg2" in data.keys():
                        seg1 = data["seg1"].to(self.device)
                        seg2 = data["seg2"].to(self.device)
                        seg1 = monai.networks.one_hot(
                            seg1, num_classes=self.num_classes
                        )
                        seg2 = monai.networks.one_hot(
                            seg2, num_classes=self.num_classes
                        )
                    elif "seg1" in data.keys():
                        seg2 = None
                        seg1 = data["seg1"].to(self.device)
                        seg1 = monai.networks.one_hot(
                            seg1, num_classes=self.num_classes
                        )
                    else:
                        assert "seg2" in data.keys()
                        seg1 = None
                        seg2 = data["seg2"].to(self.device)
                        seg2 = monai.networks.one_hot(
                            seg2, num_classes=self.num_classes
                        )

                    seg_loss = self.seg_criterion(
                        pred_seg1, pred_seg2, seg1, seg2, disp
                    )
                    seg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.segnet.parameters(), 12)
                    self.seg_optim.step()
                    seg_losses_tr.append(seg_loss.item())
                    tdata.set_postfix({"seg_loss": seg_loss.item()})
            avg_seg_loss = np.mean(seg_losses_tr, axis=0)
            self.train_losses_seg.append([epoch + 1, avg_seg_loss])
            self.logger.info(f"\tSeg Train Loss: {avg_seg_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.segnet.eval()
                seg_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_seg_val, unit="batch", desc="Validation"
                    ) as tdata:
                        for data in tdata:
                            img = data["img"].to(self.device).float()
                            label = data["seg"].to(self.device)
                            pred_seg, _ = self.segnet(img)
                            seg_loss = self.dice_loss(pred_seg, label)
                            seg_losses_val.append(seg_loss.item())
                            tdata.set_postfix({"seg_loss": seg_loss.item()})

                avg_seg_loss = np.mean(seg_losses_val, axis=0)
                self.valid_losses_seg.append([epoch + 1, avg_seg_loss])
                self.logger.info(f"\tSeg Valid Loss: {avg_seg_loss:.4f}")
                if avg_seg_loss < self.best_seg_loss:
                    self.best_seg_loss = avg_seg_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_seg_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_reg": self.regnet.state_dict(),
                            "weights_seg": self.segnet.state_dict(),
                        },
                        os.path.join(self.model_dir, "best_seg.pt"),
                    )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_seg,
                self.valid_losses_seg,
                "Segmentation_Loss",
            )
            self.seg_lr_scheduler.step()
            del (seg_loss, pred_seg1, pred_seg2, seg1, seg2, img_pair)
            torch.cuda.empty_cache()
            self.logger.info("-----" * 10)
            # Train Classifier Network
            self.swap_training(
                nets_train=[self.classnet],
                nets_freeze=[self.segnet, self.regnet],
            )
            # ---------------------------------------------------------
            #     class_net training, with seg_net and reg_net frozen
            # ---------------------------------------------------------
            class_losses_tr = []
            with tqdm(
                self.batch_gen_reg_tr(self.cfg.model.classnet.n_batch_per_epoch),
                total=self.cfg.model.classnet.n_batch_per_epoch,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.class_optim.zero_grad()
                    img_pair = data["img12"].to(self.device).float()
                    subtype = data["subtype"].to(self.device)
                    # _, reg_feats = self.regnet(img_pair)
                    _, seg_feats = self.segnet(img_pair[:, [0], :, :, :])
                    pred_class = self.classnet(seg_feats[-1])
                    class_loss = self.class_criterion(pred_class, subtype)
                    class_loss.backward()
                    self.class_optim.step()
                    class_losses_tr.append(class_loss.item())
                    tdata.set_postfix({"class_loss": class_loss.item()})

            avg_class_loss = np.mean(class_losses_tr, axis=0)
            self.train_losses_class.append([epoch + 1, avg_class_loss])
            self.logger.info(f"\tClass Train Loss: {avg_class_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.classnet.eval()
                class_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.batch_gen_reg_val(
                            self.cfg.model.classnet.n_batch_per_epoch
                        ),
                        total=self.cfg.model.classnet.n_batch_per_epoch,
                        unit="batch",
                        desc="Validation",
                    ) as tdata:
                        for data in tdata:
                            img_pair = data["img12"].to(self.device)
                            subtype = data["subtype"].to(self.device)
                            # _, reg_feats = self.regnet(img_pair)
                            _, seg_feats = self.segnet(img_pair[:, [0], :, :, :])
                            pred_class = self.classnet(seg_feats[-1])
                            class_loss = self.class_criterion(pred_class, subtype)
                            class_losses_val.append(class_loss.item())
                            tdata.set_postfix({"class_loss": class_loss.item()})
                avg_class_loss = np.mean(class_losses_val, axis=0)
                self.valid_losses_class.append([epoch + 1, avg_class_loss])
                self.logger.info(f"\tClass Valid Loss: {avg_class_loss:.4f}")
                if avg_class_loss < self.best_class_loss:
                    self.best_class_loss = avg_class_loss
                    self.logger.info(
                        f"\tSaving Best Class Model with Loss: {self.best_class_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_seg": self.segnet.state_dict(),
                            "weights_reg": self.regnet.state_dict(),
                            "weights_class": self.classnet.state_dict(),
                        },
                        os.path.join(self.model_dir, "best_class.pt"),
                    )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_class,
                self.valid_losses_class,
                "Classification_Loss",
            )
            self.class_lr_scheduler.step()
            del (
                class_loss,
                pred_class,
                # reg_feats,
                seg_feats,
                subtype,
                img_pair,
            )
            torch.cuda.empty_cache()
            self.logger.info(f"\tSaving Checkpoint for Epoch {epoch + 1}")
            self.save_checkpoint()

    def test(self):
        self.test_dir = "prediction"
        self.out_dir = os.path.join(self.test_dir, self.cfg.exp_name)
        make_if_dont_exist(self.test_dir)
        make_if_dont_exist(self.out_dir)
        self.segnet.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "best.pt"),
                map_location=torch.device("cpu"),
            )["weights_seg"]
        )
        self.segnet.to(self.device)
        self.segnet.eval()

        dc = DiceHelper(
            include_background=False, sigmoid=False, softmax=False, num_classes=2
        )
        hd = HausdorffDistanceMetric(include_background=False, reduction="mean")
        tmp = ants.image_read("../Hemo_Data_Seg/test/labels/Hem_00037_1.nii.gz")
        metrics = {}
        with torch.no_grad():
            metric_dice = []
            metric_hd = []
            for data in self.dataloader_test:
                img, seg = (
                    data["img"].unsqueeze(0).to(self.device),
                    data["seg"].unsqueeze(0).to(self.device),
                )
                filename = os.path.basename(data["seg_meta_dict"]["filename_or_obj"])
                metrics[filename] = {}
                pred, _ = self.segnet(img)
                pred = torch.argmax(pred, dim=1, keepdim=True)
                dice_score, _ = dc(pred, seg)
                hd_score = hd(pred, seg)
                metric_dice.append(dice_score.item())
                metric_hd.append(hd_score.item())
                metrics[filename]["dice"] = dice_score.item()
                metrics[filename]["hd"] = hd_score.item()
                pred_numpy = pred.squeeze().detach().cpu().numpy()
                ants_image = ants.from_numpy(
                    pred_numpy,
                    origin=tmp.origin,
                    spacing=tmp.spacing,
                    direction=tmp.direction,
                )
                ants_image.to_file(
                    os.path.join(
                        self.out_dir,
                        f"prediction_{filename}",
                    )
                )

        avg_dice = np.mean(metric_dice, axis=0)
        avg_hd = np.mean(metric_hd, axis=0)

        metrics["average"] = {
            "dice": avg_dice,
            "hd": avg_hd,
        }

        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        f.close()

    def save_checkpoint(self):
        # reg parameters
        reg_net_param = {
            "weights": self.regnet.state_dict(),
            "optimizer": self.reg_optim.state_dict(),
            "lr_scheduler": self.reg_lr_scheduler.state_dict(),
            "train_loss": self.train_losses_reg,
            "valid_loss": self.valid_losses_reg,
            "best_loss": self.best_reg_loss,
        }

        # seg parameters
        seg_net_param = {
            "weights": self.segnet.state_dict(),
            "optimizer": self.seg_optim.state_dict(),
            "lr_scheduler": self.seg_lr_scheduler.state_dict(),
            "train_loss": self.train_losses_seg,
            "valid_loss": self.valid_losses_seg,
            "best_loss": self.best_seg_loss,
        }

        # classifier parameters
        class_net_param = {
            "weights": self.classnet.state_dict(),
            "optimizer": self.class_optim.state_dict(),
            "lr_scheduler": self.class_lr_scheduler.state_dict(),
            "train_loss": self.train_losses_class,
            "valid_loss": self.valid_losses_class,
            "best_loss": self.best_class_loss,
        }

        torch.save(
            {
                "epoch": self.last_epoch_train,
                "reg": reg_net_param,
                "seg": seg_net_param,
                "class": class_net_param,
            },
            os.path.join(self.checkpoint_dir, "checkpoint.pt"),
        )

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            latest = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        else:
            raise FileNotFoundError("No checkpoint found, training from scratch")

        # epoch
        self.last_epoch_train = latest["epoch"] + 1

        # reg parameters
        reg_net_param = latest["reg"]
        self.regnet.load_state_dict(reg_net_param["weights"])
        self.reg_optim.load_state_dict(reg_net_param["optimizer"])
        self.reg_lr_scheduler.load_state_dict(reg_net_param["lr_scheduler"])
        self.train_losses_reg = reg_net_param["train_loss"]
        self.valid_losses_reg = reg_net_param["valid_loss"]
        self.best_reg_loss = reg_net_param["best_loss"]
        self.regnet.to(self.device)
        self.regnet.eval()

        # seg parameters
        seg_net_param = latest["seg"]
        self.segnet.load_state_dict(seg_net_param["weights"])
        self.seg_optim.load_state_dict(seg_net_param["optimizer"])
        self.seg_lr_scheduler.load_state_dict(seg_net_param["lr_scheduler"])
        self.train_losses_seg = seg_net_param["train_loss"]
        self.valid_losses_seg = seg_net_param["valid_loss"]
        self.best_seg_loss = seg_net_param["best_loss"]
        self.segnet.to(self.device)
        self.segnet.eval()

        # classifier parameters
        class_net_param = latest["class"]
        self.classnet.load_state_dict(class_net_param["weights"])
        self.class_optim.load_state_dict(class_net_param["optimizer"])
        self.class_lr_scheduler.load_state_dict(class_net_param["lr_scheduler"])
        self.train_losses_class = class_net_param["train_loss"]
        self.valid_losses_class = class_net_param["valid_loss"]
        self.best_class_loss = class_net_param["best_loss"]
        self.classnet.to(self.device)
        self.classnet.eval()

    def resume(self):
        self.load_checkpoint()
        self.prepare_logger(resume=True)
        self.train()
