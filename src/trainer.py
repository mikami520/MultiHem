#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 21:08:57
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-20 03:05:24
FilePath     : /MultiHem/src/trainer.py
Description  : Trainer of MultiHem
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import os
import datetime
import numpy as np
from tqdm.auto import tqdm
from src.utils import setup_logger, plot_progress, make_if_dont_exist
from src.model import SegNet, RegNet, Fusion, Classifier
from src.loss import SegLoss, RegLoss, ClassLoss, DiceLoss
import monai


class Trainer:
    def __init__(self, cfg, data_reg_tr, data_reg_val, data_seg_val, device):
        self.cfg = cfg
        self.data_reg_tr = data_reg_tr
        self.data_reg_val = data_reg_val
        self.data_seg_val = data_seg_val
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

        # Initialize the registration network
        self.regnet = RegNet(
            in_channel=self.cfg.model.regnet.in_channel,
            out_channel=self.cfg.model.regnet.out_channel,
            encode_layers=self.cfg.model.regnet.encode_layers,
            decode_layers=self.cfg.model.regnet.decode_layers,
            stride=self.cfg.model.regnet.stride,
            dropout=self.cfg.model.regnet.dropout,
            norm=self.cfg.model.regnet.norm,
        ).to(self.device)

        self.fusion = Fusion(
            seg_encode_layers=self.cfg.model.segnet.seg_encode_layers,
            reg_encode_layers=self.cfg.model.regnet.reg_encode_layers,
            seg_stride=self.cfg.model.segnet.seg_stride,
        ).to(self.device)

        self.classnet = Classifier(
            in_channel=self.cfg.model.classnet.in_channel,
            out_channel=self.cfg.model.classnet.out_channel,
        ).to(self.device)

    def prepare_optimizer(self):
        self.seg_optimizer = torch.optim.Adam(
            self.segnet.parameters(),
            lr=self.cfg.model.segnet.lr,
            weight_decay=1e-5,
        )
        self.reg_optimizer = torch.optim.Adam(
            self.regnet.parameters(),
            lr=self.cfg.model.regnet.lr,
            weight_decay=1e-5,
        )
        self.class_optimizer = torch.optim.Adam(
            self.classnet.parameters(),
            lr=self.cfg.model.classnet.lr,
            weight_decay=1e-5,
        )

    def prepare_loss(self):
        self.seg_criterion = SegLoss()
        self.reg_criterion = RegLoss(num_classes=self.num_classes)
        self.class_criterion = ClassLoss()
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
            self.logger.info(f"Start {self.cfg.experiment} Training From Scratch")
        else:
            self.logger.info(
                f"Resuming Training from Epoch {self.last_epoch_train + 1}"
            )

    def prepare_dataloader(self):
        pass

    def swap_training(self, net_train, net_freeze1, net_freeze2):
        for param in net_train.parameters():
            param.requires_grad = True

        for param in net_freeze1.parameters():
            param.requires_grad = False

        for param in net_freeze2.parameters():
            param.requires_grad = False

        net_train.train()
        net_freeze1.eval()
        net_freeze2.eval()

    def train(self):
        self.logger.info("----------------------------------")
        self.logger.info("You are using alternate training mode")
        self.logger.info("----------------------------------")
        for epoch in range(self.last_epoch_train, self.cfg.train.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.cfg.train.epochs}")
            reg_losses_tr = []
            self.logger.info("----" * 10)
            self.swap_training(self.regnet, self.segnet, self.classnet)
            # ---------------------------------------------------------
            #     reg_net training, with seg_net and class_net frozen
            # ---------------------------------------------------------
            with tqdm(
                self.dataloader_reg_tr,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.reg_optimizer.zero_grad()
                    img_pair = data["img12"]
                    img_pair = img_pair.to(self.device)
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
                    self.reg_optimizer.step()
                    reg_losses_tr.append(reg_loss.item())
                    tdata.set_postfix({"reg_loss": reg_loss.item()})

            avg_reg_loss = np.mean(reg_losses_tr, axis=0)
            self.train_losses_reg.append([epoch + 1, avg_reg_loss])
            self.logger.info(f"Reg Train Loss: {avg_reg_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.regnet.eval()
                reg_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_reg_val, unit="batch", desc="Validation"
                    ) as tdata:
                        for data in tdata:
                            img_pair = data["img12"]
                            img_pair = img_pair.to(self.device)
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
                self.logger.info(f"Reg Valid Loss: {avg_reg_loss:.4f}")
                if avg_reg_loss < self.best_reg_loss:
                    self.best_reg_loss = avg_reg_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_reg_loss:.4f}"
                    )
                    torch.save(
                        self.regnet.state_dict(),
                        os.path.join(self.model_dir, "best_reg.pt"),
                    )
            plot_progress(
                self.logger,
                os.path.join(self.plot_dir, "reg_loss.png"),
                self.train_losses_reg,
                self.valid_losses_reg,
                "Registration Loss",
            )
            del (reg_loss, disp, seg1, seg2, img_pair)
            torch.cuda.empty_cache()

            self.logger.info("----" * 10)
            self.swap_training(self.segnet, self.regnet, self.classnet)
            # ---------------------------------------------------------
            #     seg_net training, with reg_net and class_net frozen
            # ---------------------------------------------------------
            seg_losses_tr = []
            with tqdm(
                self.dataloader_seg_tr,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.seg_optimizer.zero_grad()
                    img_pair = data["img12"]
                    img_pair = img_pair.to(self.device)
                    disp, _ = self.regnet(img_pair)
                    pred_seg1, _ = self.segnet(img_pair[:, [0], :, :, :])
                    pred_seg2, _ = self.segnet(img_pair[:, [1], :, :, :])
                    pred_seg1 = torch.softmax(pred_seg1, dim=1)
                    pred_seg2 = torch.softmax(pred_seg2, dim=1)
                    if "seg1" in data.keys() and "seg2" in data.keys():
                        seg1 = data["seg1"]
                        seg1 = seg1.to(self.device)
                        seg2 = data["seg2"]
                        seg2 = seg2.to(self.device)
                        seg1 = monai.networks.one_hot(
                            seg1, num_classes=self.num_classes
                        )
                        seg2 = monai.networks.one_hot(
                            seg2, num_classes=self.num_classes
                        )
                    elif "seg1" in data.keys():
                        seg2 = None
                        seg1 = data["seg1"]
                        seg1 = seg1.to(self.device)
                        seg1 = monai.networks.one_hot(
                            seg1, num_classes=self.num_classes
                        )
                    else:
                        assert "seg2" in data.keys()
                        seg1 = None
                        seg2 = data["seg2"]
                        seg2 = seg2.to(self.device)
                        seg2 = monai.networks.one_hot(
                            seg2, num_classes=self.num_classes
                        )

                    seg_loss = self.seg_criterion(
                        pred_seg1, pred_seg2, seg1, seg2, disp
                    )
                    seg_loss.backward()
                    self.seg_optimizer.step()
                    seg_losses_tr.append(seg_loss.item())
                    tdata.set_postfix({"seg_loss": seg_loss.item()})
            avg_seg_loss = np.mean(seg_losses_tr, axis=0)
            self.train_losses_seg.append([epoch + 1, avg_seg_loss])
            self.logger.info(f"Seg Train Loss: {avg_seg_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.segnet.eval()
                seg_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_seg_val, unit="batch", desc="Validation"
                    ) as tdata:
                        for data in tdata:
                            img = data["img"]
                            img = img.to(self.device)
                            label = data["label"]
                            label = label.to(self.device)
                            pred_seg, _ = self.segnet(img)
                            seg_loss = self.dice_loss(pred_seg, label)
                            seg_losses_val.append(seg_loss.item())
                            tdata.set_postfix({"seg_loss": seg_loss.item()})

                avg_seg_loss = np.mean(seg_losses_val, axis=0)
                self.valid_losses_seg.append([epoch + 1, avg_seg_loss])
                self.logger.info(f"Seg Valid Loss: {avg_seg_loss:.4f}")
                if avg_seg_loss < self.best_seg_loss:
                    self.best_seg_loss = avg_seg_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_seg_loss:.4f}"
                    )
                    torch.save(
                        self.segnet.state_dict(),
                        os.path.join(self.model_dir, "best_seg.pt"),
                    )
            plot_progress(
                self.logger,
                os.path.join(self.plot_dir, "seg_loss.png"),
                self.train_losses_seg,
                self.valid_losses_seg,
                "Segmentation Loss",
            )
            del (seg_loss, pred_seg1, pred_seg2, seg1, seg2, img_pair)
            torch.cuda.empty_cache()

            self.logger.info("----" * 10)
            self.swap_training(self.classnet, self.segnet, self.regnet)
            # ---------------------------------------------------------
            #     class_net training, with seg_net and reg_net frozen
            # ---------------------------------------------------------
            class_losses_tr = []

            with tqdm(
                self.dataloader_class_tr,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.class_optimizer.zero_grad()
                    img_pair = data["img12"]
                    label1 = data["label1"]
                    label2 = data["label2"]
                    img_pair = img_pair.to(self.device)
                    label1 = label1.to(self.device)
                    label2 = label2.to(self.device)
                    _, reg_feats = self.regnet(img_pair)
                    _, seg_feats1 = self.segnet(img_pair[:, [0], :, :, :])
                    _, seg_feats2 = self.segnet(img_pair[:, [1], :, :, :])
                    fused_feats1 = self.fusion(seg_feats1, reg_feats)
                    fused_feats2 = self.fusion(seg_feats2, reg_feats)
                    pred_class1 = self.classnet(fused_feats1)
                    pred_class2 = self.classnet(fused_feats2)
                    class_loss1 = self.class_criterion(pred_class1, label1)
                    class_loss2 = self.class_criterion(pred_class2, label2)
                    class_loss = class_loss1 + class_loss2
                    class_loss.backward()
                    self.class_optimizer.step()
                    class_losses_tr.append(class_loss.item())
                    tdata.set_postfix({"class_loss": class_loss.item()})
            avg_class_loss = np.mean(class_losses_tr, axis=0)
            self.train_losses_class.append([epoch + 1, avg_class_loss])
            self.logger.info(f"Class Train Loss: {avg_class_loss:.4f}")

            if (epoch + 1) % self.cfg.train.val_iter == 0:
                self.classnet.eval()
                class_losses_val = []
                with torch.no_grad():
                    with tqdm(
                        self.dataloader_class_val, unit="batch", desc="Validation"
                    ) as tdata:
                        for data in tdata:
                            img_pair = data["img12"]
                            label1 = data["label1"]
                            label2 = data["label2"]
                            img_pair = img_pair.to(self.device)
                            label1 = label1.to(self.device)
                            label2 = label2.to(self.device)
                            _, reg_feats = self.regnet(img_pair)
                            _, seg_feats1 = self.segnet(img_pair[:, [0], :, :, :])
                            _, seg_feats2 = self.segnet(img_pair[:, [1], :, :, :])
                            fused_feats1 = self.fusion(seg_feats1, reg_feats)
                            fused_feats2 = self.fusion(seg_feats2, reg_feats)
                            pred_class1 = self.classnet(fused_feats1)
                            pred_class2 = self.classnet(fused_feats2)
                            class_loss1 = self.class_criterion(pred_class1, label1)
                            class_loss2 = self.class_criterion(pred_class2, label2)
                            class_loss = class_loss1 + class_loss2
                            class_losses_val.append(class_loss.item())
                            tdata.set_postfix({"class_loss": class_loss.item()})
                avg_class_loss = np.mean(class_losses_val, axis=0)
                self.valid_losses_class.append([epoch + 1, avg_class_loss])
                self.logger.info(f"Class Valid Loss: {avg_class_loss:.4f}")
                if avg_class_loss < self.best_class_loss:
                    self.best_class_loss = avg_class_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_class_loss:.4f}"
                    )
                    torch.save(
                        self.classnet.state_dict(),
                        os.path.join(self.model_dir, "best_class.pt"),
                    )
            plot_progress(
                self.logger,
                os.path.join(self.plot_dir, "class_loss.png"),
                self.train_losses_class,
                self.valid_losses_class,
                "Classification Loss",
            )
            del (
                class_loss,
                class_loss1,
                class_loss2,
                pred_class1,
                pred_class2,
                label1,
                label2,
                img_pair,
            )
            torch.cuda.empty_cache()

            self.save_checkpoint()

    def save_checkpoint(self):
        # reg parameters
        reg_net_param = {
            "weights": self.regnet.state_dict(),
            "optimizer": self.reg_optimizer.state_dict(),
            "train_loss": self.train_losses_reg,
            "valid_loss": self.valid_losses_reg,
            "best_loss": self.best_reg_loss,
        }

        # seg parameters
        seg_net_param = {
            "weights": self.segnet.state_dict(),
            "optimizer": self.seg_optimizer.state_dict(),
            "train_loss": self.train_losses_seg,
            "valid_loss": self.valid_losses_seg,
            "best_loss": self.best_seg_loss,
        }

        # classifier parameters
        class_net_param = {
            "weights": self.classnet.state_dict(),
            "optimizer": self.class_optimizer.state_dict(),
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
        latest = (
            torch.load(
                os.path.join(self.checkpoint_dir, "checkpoint.pt"),
                map_location=self.device,
            )
            if os.path.exists(os.path.join(self.checkpoint_dir, "checkpoint.pt"))
            else None
        )
        if latest is None:
            print("No checkpoint found, training from scratch")
            return

        # epoch
        self.last_epoch_train = latest["epoch"] + 1

        # reg parameters
        reg_net_param = latest["reg"]
        self.regnet.load_state_dict(reg_net_param["weights"])
        self.reg_optimizer.load_state_dict(reg_net_param["optimizer"])
        self.train_losses_reg = reg_net_param["train_loss"]
        self.valid_losses_reg = reg_net_param["valid_loss"]
        self.best_reg_loss = reg_net_param["best_loss"]

        # seg parameters
        seg_net_param = latest["seg"]
        self.segnet.load_state_dict(seg_net_param["weights"])
        self.seg_optimizer.load_state_dict(seg_net_param["optimizer"])
        self.train_losses_seg = seg_net_param["train_loss"]
        self.valid_losses_seg = seg_net_param["valid_loss"]
        self.best_seg_loss = seg_net_param["best_loss"]

        # classifier parameters
        class_net_param = latest["class"]
        self.classnet.load_state_dict(class_net_param["weights"])
        self.class_optimizer.load_state_dict(class_net_param["optimizer"])
        self.train_losses_class = class_net_param["train_loss"]
        self.valid_losses_class = class_net_param["valid_loss"]
        self.best_class_loss = class_net_param["best_loss"]

        self.segnet.eval()
        self.regnet.eval()
        self.classnet.eval()

    def resume(self):
        self.load_checkpoint()
        self.prepare_logger(resume=True)
        self.train()

    def test(self, *args, **kwargs):
        pass
