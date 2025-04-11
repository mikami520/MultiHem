#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-04-04 13:29:40
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-10 13:34:05
FilePath     : /Downloads/MultiHem/src/trainer.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn.functional as F
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
                keys=["img", "seg"],
                image_only=True,
            ),
            EnsureChannelFirstd(keys=["img", "seg"]),
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
            RandRotated(
                keys=["img", "seg"],
                range_z=np.pi / 12,
                mode=["bilinear", "nearest"],
                prob=0.5,
                keep_size=True,
            ),
            RandZoomd(
                keys=["img", "seg"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
                mode=["trilinear", "nearest"],
                keep_size=True,
            ),
        ]
    ),
    "valid": Compose(
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


class Trainer:
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

        self.classnet = Classifier(
            in_channels=self.cfg.model.classnet.in_channel,
            num_classes=self.cfg.model.classnet.out_channel,
            res_channels=self.cfg.model.classnet.res_channel,
            dropout_p=self.cfg.model.classnet.dropout,
            deep_supervision=False,
        ).to(self.device)

        self.segnet.eval()
        self.classnet.eval()

    def prepare_optimizer(self):
        # Create separate optimizers for each network
        self.seg_optim = torch.optim.AdamW(
            self.segnet.parameters(),
            lr=self.cfg.model.segnet.lr,
            weight_decay=self.cfg.model.segnet.weight_decay,
        )

        self.class_optim = torch.optim.AdamW(
            self.classnet.parameters(),
            lr=self.cfg.model.classnet.lr,
            weight_decay=self.cfg.model.classnet.weight_decay,
        )

        # Separate schedulers if needed
        self.seg_lr_scheduler = CosineAnnealingLR(
            self.seg_optim, T_max=self.cfg.train.epochs, eta_min=1e-6
        )
        self.class_lr_scheduler = CosineAnnealingLR(
            self.class_optim, T_max=self.cfg.train.epochs, eta_min=1e-6
        )

    def prepare_loss(self):
        self.seg_criterion = DiceLoss()
        self.class_criterion = ClassLoss(label_smoothing=0.1)
        self.train_losses_total = []
        self.valid_losses_total = []
        self.train_losses_seg = []
        self.valid_losses_seg = []
        self.train_losses_class = []
        self.valid_losses_class = []
        self.best_total_loss = np.inf

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
            train_data, valid_data = partition_dataset(
                data_path,
                ratios=(8, 2),
                shuffle=True,
            )
            dataset_train = CacheDataset(
                data=train_data,
                transform=TRANSFORMS["train"],
                cache_num=16,
            )
            dataset_val = CacheDataset(
                data=valid_data,
                transform=TRANSFORMS["valid"],
                cache_num=16,
            )
            self.dataloader_train = DataLoader(
                dataset_train,
                batch_size=self.cfg.model.segnet.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=self.cfg.model.segnet.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        else:
            self.datatest_test = CacheDataset(
                data=data_path,
                transform=TRANSFORMS["valid"],
                cache_num=16,
            )

    def train(self):
        for epoch in range(self.last_epoch_train, self.cfg.train.epochs):
            self.logger.info(f"\tEpoch {epoch + 1}/{self.cfg.train.epochs}")
            self.last_epoch_train = epoch
            total_losses_tr = []
            seg_losses_tr = []
            class_losses_tr = []
            self.segnet.train()
            self.classnet.train()

            # Configuration for gradient accumulation
            accumulation_steps = 4  # For classification network
            batch_counter = 0
            accumulated_class_loss = 0

            with tqdm(
                self.dataloader_train,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.seg_optim.zero_grad()

                    # Only zero classnet gradients at the start of accumulation cycle
                    if batch_counter % accumulation_steps == 0:
                        self.class_optim.zero_grad()

                    img, seg, subtype = (
                        data["img"].to(self.device).float(),
                        data["seg"].to(self.device),
                        data["subtype"].to(self.device),
                    )
                    pred_seg, seg_feats = self.segnet(img)
                    mid_feat = F.adaptive_avg_pool3d(
                        seg_feats[-3].detach(), (2, 2, 2)
                    )  # [B, 64, 2, 2, 2]
                    high_feat = F.adaptive_avg_pool3d(
                        seg_feats[-1].detach(), (2, 2, 2)
                    )  # [B, 256, 2, 2, 2]
                    cls_input = torch.cat(
                        [mid_feat, high_feat], dim=1
                    )  # [B, 320, 2, 2, 2]
                    pred_class = self.classnet(cls_input)

                    seg_loss = self.seg_criterion(
                        pred_seg,
                        seg,
                    )
                    class_loss = (
                        self.class_criterion(
                            pred_class,
                            subtype,
                        )
                        / accumulation_steps
                    )

                    seg_loss.backward()
                    class_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.segnet.parameters(), 12.0)
                    self.seg_optim.step()
                    # Track the unscaled classification loss for logging
                    actual_class_loss = class_loss.item() * accumulation_steps
                    accumulated_class_loss += actual_class_loss

                    # Update classnet only after accumulation_steps iterations
                    if (batch_counter + 1) % accumulation_steps == 0:
                        self.class_optim.step()

                        # Log the average accumulated class loss
                        avg_class_loss = accumulated_class_loss / accumulation_steps
                        class_losses_tr.append(avg_class_loss)
                        accumulated_class_loss = 0  # Reset accumulator

                    seg_losses_tr.append(seg_loss.item())
                    # Log combined loss when we update both networks
                    if (batch_counter + 1) % accumulation_steps == 0:
                        total_loss = seg_loss.item() + avg_class_loss
                        total_losses_tr.append(total_loss)

                        tdata.set_postfix(
                            {
                                "total_loss": total_loss,
                                "seg_loss": seg_loss.item(),
                                "class_loss": avg_class_loss,
                            }
                        )
                    else:
                        # Show accumulation progress
                        tdata.set_postfix(
                            {
                                "seg_loss": seg_loss.item(),
                                "class_acc": f"{batch_counter % accumulation_steps + 1}/{accumulation_steps}",
                            }
                        )

                    batch_counter += 1

            avg_total_loss = np.mean(total_losses_tr, axis=0)
            avg_seg_loss = np.mean(seg_losses_tr, axis=0)
            avg_class_loss = np.mean(class_losses_tr, axis=0)
            self.train_losses_total.append([epoch + 1, avg_total_loss])
            self.train_losses_seg.append([epoch + 1, avg_seg_loss])
            self.train_losses_class.append([epoch + 1, avg_class_loss])
            self.logger.info(
                f"\tTotal Train Loss: {avg_total_loss:.4f}, Seg Train Loss: {avg_seg_loss:.4f}, Class Train Loss: {avg_class_loss:.4f}"
            )

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.segnet.eval()
                self.classnet.eval()
                total_losses_val = []
                seg_losses_val = []
                class_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_val, unit="batch", desc="Validation"
                    ) as tdata:
                        for data in tdata:
                            img, label, subtype = (
                                data["img"].to(self.device).float(),
                                data["seg"].to(self.device),
                                data["subtype"].to(self.device),
                            )
                            pred_seg, seg_feats = self.segnet(img)
                            mid_feat = F.adaptive_avg_pool3d(
                                seg_feats[-3].detach(), (2, 2, 2)
                            )  # [B, 64, 2, 2, 2]
                            high_feat = F.adaptive_avg_pool3d(
                                seg_feats[-1].detach(), (2, 2, 2)
                            )  # [B, 256, 2, 2, 2]
                            cls_input = torch.cat(
                                [mid_feat, high_feat], dim=1
                            )  # [B, 320, 2, 2, 2]
                            pred_class = self.classnet(cls_input)
                            seg_loss = self.seg_criterion(pred_seg, label)
                            class_loss = self.class_criterion(pred_class, subtype)
                            total_loss = seg_loss + class_loss
                            total_losses_val.append(total_loss.item())
                            seg_losses_val.append(seg_loss.item())
                            class_losses_val.append(class_loss.item())
                            tdata.set_postfix(
                                {
                                    "total_loss": total_loss.item(),
                                    "seg_loss": seg_loss.item(),
                                    "class_loss": class_loss.item(),
                                }
                            )
                avg_total_loss = np.mean(total_losses_val, axis=0)
                avg_seg_loss = np.mean(seg_losses_val, axis=0)
                avg_class_loss = np.mean(class_losses_val, axis=0)
                self.valid_losses_total.append([epoch + 1, avg_total_loss])
                self.valid_losses_seg.append([epoch + 1, avg_seg_loss])
                self.valid_losses_class.append([epoch + 1, avg_class_loss])
                self.logger.info(
                    f"\tTotal Valid Loss: {avg_total_loss:.4f}, Seg Valid Loss: {avg_seg_loss:.4f}, Class Valid Loss: {avg_class_loss:.4f}"
                )
                if avg_total_loss < self.best_total_loss:
                    self.best_total_loss = avg_total_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_total_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_seg": self.segnet.state_dict(),
                            "weights_classg": self.classnet.state_dict(),
                        },
                        os.path.join(self.model_dir, "best.pt"),
                    )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_total,
                self.valid_losses_total,
                "Total_Loss",
            )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_seg,
                self.valid_losses_seg,
                "Segmentation_Loss",
            )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_class,
                self.valid_losses_class,
                "Classification_Loss",
            )
            self.seg_lr_scheduler.step()
            self.class_lr_scheduler.step()
            del (
                seg_loss,
                class_loss,
                total_loss,
                img,
                seg,
                subtype,
                pred_seg,
                pred_class,
                cls_input,
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
            include_background=True,
            sigmoid=False,
            softmax=False,
            num_classes=2,
            reduction="mean",
        )
        hd = HausdorffDistanceMetric(include_background=True, reduction="mean")
        tmp = ants.image_read("../Hemo_Data_Seg/test/labels/Hem_00037_1.nii.gz")
        metrics = {}
        with torch.no_grad():
            metric_dice = []
            metric_hd = []
            for data in self.datatest_test:
                img, seg = (
                    data["img"].unsqueeze(0).to(self.device),
                    data["seg"].unsqueeze(0).to(self.device),
                )
                seg = monai.networks.utils.one_hot(seg, num_classes=self.num_classes)
                filename = os.path.basename(data["seg_meta_dict"]["filename_or_obj"])
                metrics[filename] = {}
                pred, _ = self.segnet(img)
                pred = torch.argmax(pred, dim=1, keepdim=True)
                pred = monai.networks.utils.one_hot(pred, num_classes=self.num_classes)
                dice_score, _ = dc(pred, seg)
                hd_score = hd(pred, seg)
                hd_score = torch.mean(hd_score)
                metric_dice.append(dice_score.item())
                metric_hd.append(hd_score.item())
                metrics[filename]["dice"] = dice_score.item()
                metrics[filename]["hd"] = hd_score.item()
                pred_numpy = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
                print(pred_numpy.shape)
                exit()
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
        # seg parameters
        seg_net_param = {
            "weights": self.segnet.state_dict(),
            "optimizer": self.seg_optim.state_dict(),
            "lr_scheduler": self.seg_lr_scheduler.state_dict(),
            "train_loss": self.train_losses_seg,
            "valid_loss": self.valid_losses_seg,
        }

        # classifier parameters
        class_net_param = {
            "weights": self.classnet.state_dict(),
            "optimizer": self.class_optim.state_dict(),
            "lr_scheduler": self.class_lr_scheduler.state_dict(),
            "train_loss": self.train_losses_class,
            "valid_loss": self.valid_losses_class,
        }

        torch.save(
            {
                "epoch": self.last_epoch_train,
                "train_loss": self.train_losses_total,
                "valid_loss": self.valid_losses_total,
                "best_loss": self.best_total_loss,
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
        self.best_total_loss = latest["best_loss"]
        self.train_losses_total = latest["train_loss"]
        self.valid_losses_total = latest["valid_loss"]

        # seg parameters
        seg_net_param = latest["seg"]
        self.segnet.load_state_dict(seg_net_param["weights"])
        self.seg_optim.load_state_dict(seg_net_param["optimizer"])
        self.seg_lr_scheduler.load_state_dict(seg_net_param["lr_scheduler"])
        self.train_losses_seg = seg_net_param["train_loss"]
        self.valid_losses_seg = seg_net_param["valid_loss"]
        self.segnet.to(self.device)
        self.segnet.eval()

        # classifier parameters
        class_net_param = latest["class"]
        self.classnet.load_state_dict(class_net_param["weights"])
        self.class_optim.load_state_dict(class_net_param["optimizer"])
        self.class_lr_scheduler.load_state_dict(class_net_param["lr_scheduler"])
        self.train_losses_class = class_net_param["train_loss"]
        self.valid_losses_class = class_net_param["valid_loss"]
        self.classnet.to(self.device)
        self.classnet.eval()

    def resume(self):
        self.load_checkpoint()
        self.prepare_logger(resume=True)
        self.train()
