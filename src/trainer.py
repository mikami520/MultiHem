#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-19 21:08:57
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-03-09 04:55:15
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
from src.utils import (
    setup_logger,
    plot_progress,
    make_if_dont_exist,
    DataHandler,
    create_batch_generator,
)
from src.model import SegNet, RegNet, Fusion, Classifier
from src.loss import SegLoss, RegLoss, ClassLoss, DiceLoss
import monai

TRANSFORMS = {
    "seg": monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=["img", "seg"], image_only=False),
            monai.transforms.TransposeD(keys=["img", "seg"], indices=(2, 1, 0)),
            monai.transforms.EnsureChannelFirstD(keys=["img", "seg"]),
        ]
    ),
    "both": monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(
                keys=["img1", "seg1", "img2", "seg2"],
                image_only=False,
                allow_missing_keys=True,
            ),
            monai.transforms.TransposeD(
                keys=["img1", "seg1", "img2", "seg2"],
                indices=(2, 1, 0),
                allow_missing_keys=True,
            ),
            monai.transforms.EnsureChannelFirstD(
                keys=["img1", "seg1", "img2", "seg2"], allow_missing_keys=True
            ),
            monai.transforms.ConcatItemsD(keys=["img1", "img2"], name="img12", dim=0),
            monai.transforms.DeleteItemsD(keys=["img1", "img2"]),
        ]
    ),
}


class Trainer:
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
            seg_encode_layers=self.cfg.model.segnet.encode_layers,
            reg_encode_layers=self.cfg.model.regnet.encode_layers,
            stride=self.cfg.model.segnet.stride,
        ).to(self.device)

        self.classnet = Classifier(
            in_channel=self.cfg.model.classnet.in_channel,
            num_classes=self.cfg.model.classnet.out_channel,
            hidden_dim=self.cfg.model.classnet.hidden_dim,
            num_transformer_layers=self.cfg.model.classnet.num_transformer_layers,
            num_heads=self.cfg.model.classnet.num_heads,
            dropout=self.cfg.model.classnet.dropout,
        ).to(self.device)

        self.segnet.eval()
        self.regnet.eval()
        self.fusion.eval()
        self.classnet.eval()

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
        self.class_optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.fusion.parameters(),
                    "lr": self.cfg.model.fusion.lr,
                    "weight_decay": self.cfg.model.fusion.weight_decay,
                },
                {
                    "params": self.classnet.parameters(),
                    "lr": self.cfg.model.classnet.lr,
                    "weight_decay": self.cfg.model.classnet.weight_decay,
                },
            ]
        )

    def prepare_loss(self):
        w_super, _ = self.cfg.model.segnet.loss_weights
        w_sim, w_penal, w_ana = self.cfg.model.regnet.loss_weights
        self.seg_criterion = SegLoss(w_super, w_ana)
        self.reg_criterion = RegLoss(
            w_sim, w_penal, w_ana, num_classes=self.num_classes
        )
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
            self.logger.info(f"\tStart {self.cfg.experiment} Training From Scratch")
        else:
            self.logger.info(
                f"\tResuming Training from Epoch {self.last_epoch_train + 1}"
            )

    def prepare_dataloader(self):
        handler = DataHandler(self.cfg)
        data_path = handler._get_paths()

        data_seg_avail = list(filter(lambda x: "seg" in x.keys(), data_path))
        data_seg_not_avail = list(filter(lambda x: "seg" not in x.keys(), data_path))

        data_seg_avail_tr, data_seg_avail_val = monai.data.utils.partition_dataset(
            data_seg_avail, ratios=(8, 2)
        )

        self.dataloader_seg_val = monai.data.DataLoader(
            monai.data.CacheDataset(
                data=data_seg_avail_val,
                transform=TRANSFORMS["seg"],
                cache_num=16,
            ),
            batch_size=int(self.cfg.model.segnet.batch_size) * 2,
            num_workers=2,
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
                data=data_list, transform=TRANSFORMS["both"], cache_num=32
            )
            for seg_availability, data_list in subdivided_data_both_tr.items()
        }
        subdivided_dataset_both_val = {
            seg_availability: monai.data.CacheDataset(
                data=data_list, transform=TRANSFORMS["both"], cache_num=32
            )
            for seg_availability, data_list in subdivided_data_both_val.items()
        }

        dataloader_reg_tr = {
            seg_availability: (
                monai.data.DataLoader(
                    dataset,
                    batch_size=int(self.cfg.model.regnet.batch_size),
                    num_workers=2,
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
                    batch_size=int(self.cfg.model.regnet.batch_size) * 2,
                    num_workers=2,
                    shuffle=True,
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

    # New helper: accepts lists of networks to train vs. freeze.
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
        self.logger.info("-----" * 10)
        self.logger.info("\tYou are using alternate training mode")
        self.logger.info("-----" * 10)
        for epoch in range(self.last_epoch_train, self.cfg.train.epochs):
            self.logger.info(f"\tEpoch {epoch + 1}/{self.cfg.train.epochs}")
            self.last_epoch_train = epoch
            reg_losses_tr = []
            self.logger.info("-----" * 10)
            self.swap_training([self.regnet], [self.segnet, self.fusion, self.classnet])
            # ---------------------------------------------------------
            #     reg_net training, with seg_net and class_net frozen
            # ---------------------------------------------------------
            with tqdm(
                self.batch_gen_reg_tr(self.cfg.model.regnet.n_batch_per_epoch),
                total=self.cfg.model.regnet.n_batch_per_epoch,
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
                self.logger.info(f"\tReg Valid Loss: {avg_reg_loss:.4f}")
                if avg_reg_loss < self.best_reg_loss:
                    self.best_reg_loss = avg_reg_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_reg_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_reg": self.regnet.state_dict(),
                            "weights_seg": self.segnet.state_dict(),
                            "weights_fusion": self.fusion.state_dict(),
                            "weights_classifier": self.classnet.state_dict(),
                        },
                        os.path.join(self.model_dir, "best_reg.pt"),
                    )
            plot_progress(
                self.logger,
                self.plot_dir,
                self.train_losses_reg,
                self.valid_losses_reg,
                "Registration_Loss",
            )
            del (reg_loss, disp, seg1, seg2, img_pair)
            torch.cuda.empty_cache()

            self.logger.info("-----" * 10)
            self.swap_training([self.segnet], [self.regnet, self.fusion, self.classnet])
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
            self.logger.info(f"\tSeg Train Loss: {avg_seg_loss:.4f}")

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
                            label = data["seg"]
                            label = label.to(self.device)
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
                            "weights_fusion": self.fusion.state_dict(),
                            "weights_classifier": self.classnet.state_dict(),
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
            del (seg_loss, pred_seg1, pred_seg2, seg1, seg2, img_pair)
            torch.cuda.empty_cache()

            self.logger.info("-----" * 10)
            # ---------------------------------------------------------
            #     fusion and class_net training, with seg_net and reg_net frozen
            # ---------------------------------------------------------
            self.swap_training([self.fusion, self.classnet], [self.segnet, self.regnet])
            class_losses_tr = []

            with tqdm(
                self.batch_gen_reg_tr(self.cfg.model.classnet.n_batch_per_epoch),
                total=self.cfg.model.classnet.n_batch_per_epoch,
                unit="batch",
                desc="Training",
            ) as tdata:
                for data in tdata:
                    self.class_optimizer.zero_grad()
                    img_pair = data["img12"].to(self.device)
                    subtype = data["subtype"].to(self.device)
                    _, reg_feats = self.regnet(img_pair)
                    _, seg_feats = self.segnet(img_pair[:, [0], :, :, :])
                    fused_feats = self.fusion(seg_feats, reg_feats)
                    pred_class = self.classnet(fused_feats)
                    class_loss = self.class_criterion(pred_class, subtype)
                    class_loss.backward()
                    self.class_optimizer.step()
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
                            img_pair = data["img12"]
                            subtype = data["subtype"]
                            img_pair = img_pair.to(self.device)
                            subtype = subtype.to(self.device)
                            _, reg_feats = self.regnet(img_pair)
                            _, seg_feats = self.segnet(img_pair[:, [0], :, :, :])
                            fused_feats = self.fusion(seg_feats, reg_feats)
                            pred_class = self.classnet(fused_feats)
                            class_loss = self.class_criterion(pred_class, subtype)
                            class_losses_val.append(class_loss.item())
                            tdata.set_postfix({"class_loss": class_loss.item()})
                avg_class_loss = np.mean(class_losses_val, axis=0)
                self.valid_losses_class.append([epoch + 1, avg_class_loss])
                self.logger.info(f"\tClass Valid Loss: {avg_class_loss:.4f}")
                if avg_class_loss < self.best_class_loss:
                    self.best_class_loss = avg_class_loss
                    self.logger.info(
                        f"\tSaving Best Model with Loss: {self.best_class_loss:.4f}"
                    )
                    torch.save(
                        {
                            "weights_reg": self.regnet.state_dict(),
                            "weights_seg": self.segnet.state_dict(),
                            "weights_fusion": self.fusion.state_dict(),
                            "weights_classifier": self.classnet.state_dict(),
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
            del (
                class_loss,
                pred_class,
                subtype,
                img_pair,
            )
            torch.cuda.empty_cache()
            self.logger.info(f"\tSaving Checkpoint for Epoch {epoch + 1}")
            self.save_checkpoint()

    def test(self, *args, **kwargs):
        # TODO: Implement inference step
        pass

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
            "weights_fusion": self.fusion.state_dict(),
            "weights_classifier": self.classnet.state_dict(),
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
                weights_only=False,
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
        self.fusion.load_state_dict(class_net_param["weights_fusion"])
        self.classnet.load_state_dict(class_net_param["weights_classifier"])
        self.class_optimizer.load_state_dict(class_net_param["optimizer"])
        self.train_losses_class = class_net_param["train_loss"]
        self.valid_losses_class = class_net_param["valid_loss"]
        self.best_class_loss = class_net_param["best_loss"]

        self.segnet.eval()
        self.regnet.eval()
        self.fusion.eval()
        self.classnet.eval()

    def resume(self):
        self.load_checkpoint()
        self.prepare_logger(resume=True)
        self.train()
