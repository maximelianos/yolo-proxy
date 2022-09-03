import sys
import argparse
from glob import glob
from pathlib import Path
import time
import os
import datetime as dt

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision.models import resnet18

import tqdm
import matplotlib.pyplot as plt

from dataset import BaseDataset, TrainDataset


DEVICE = "cuda"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512 * 3, 1)

    def forward(self, ref, dist):
        b, c, h, w = ref.shape
        a = self.backbone(ref)  # (b, c, h, w) -> (b, 512)
        b = self.backbone(dist)
        concat = torch.cat((a, b, a-b), dim=1)  # (b, 512*3)
        x = self.fc(concat)  # (b, 1)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()

        # Extract ResNet18 features
        self.pretrained = resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.pretrained._modules.pop(self.layers[-i])
        self.feat_extractor = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

        self.fc = nn.Linear(512 * 2 * 2 * 3, 1)

    def forward(self, img_ref, img_dist, padded_bboxes, padded_ious):
        feat_ref = self.feat_extractor(img_ref) # b 512 h w
        feat_dist = self.feat_extractor(img_dist)

        bboxes = []  # [ [x1 y1 x2 y2], ... ]
        ious = []  # [ i1 i2 ... ]

        b, k, _coors = padded_bboxes.shape
        for ibatch in range(b):
            for ibox in range(k):
                box = padded_bboxes[ibatch, ibox]
                if box.sum() > 1:
                    bboxes.append(torch.tensor([ibatch] + list(box), dtype=torch.float))
                    ious.append(padded_ious[ibatch, ibox])
        bboxes = torch.stack(bboxes).to(feat_ref) # (n_boxes, 5)
        ious = torch.tensor(ious).to(feat_ref) # (n_boxes)
        n_bboxes = bboxes.shape[0]

        b, c, h, w = img_ref.shape
        _b, _c, fh, fw = feat_ref.shape
        pooling_size = (2, 2)
        roi_ref = torchvision.ops.roi_pool(feat_ref, boxes=bboxes, output_size=pooling_size, spatial_scale=fh / h) # (n_boxes, c, gridh, gridw)
        roi_dist = torchvision.ops.roi_pool(feat_dist, boxes=bboxes, output_size=pooling_size, spatial_scale=fh / h)
        roi_diff = torchvision.ops.roi_pool(feat_ref - feat_dist, boxes=bboxes, output_size=pooling_size, spatial_scale=fh / h)
        concat = torch.stack((roi_ref, roi_dist, roi_diff), dim=1)
        flat = concat.view(n_bboxes, -1)
        out = self.fc(flat)

        return [out, ious]

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint_path):
    best_spearman = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        # Train

        model.train()
        val_loss = []
        outs_val = []
        gt_val = []
        step = 0
        for data in tqdm.tqdm(train_loader):
            if step > 50:
                break
            step += 1
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reference = data["reference"].to(DEVICE) # shape (b c h w)
                distorted = data["distorted"].to(DEVICE) # shape (b c h w)
                padded_bboxes = data["bbox"].to(DEVICE)  # shape (b, k, 4)
                padded_ious = data["iou_dist"].to(DEVICE) # shape (b)
                outputs, labels = model(reference, distorted, padded_bboxes, padded_ious) # shape (b 1)
                loss = criterion(outputs, labels[:, None]) # add dimension at the end
                # print("L1 (MSE):", loss.item())
            val_loss.append(loss.item())
            outs_val.extend(list(outputs[:, 0].detach().cpu().numpy()))
            gt_val.extend(list(labels.detach().cpu().numpy()))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        mean_loss = np.mean(val_loss)
        corr_spearman = spearmanr(outs_val, gt_val)[0]
        corr_pearsonr = pearsonr(outs_val, gt_val)[0]
        corr_kendall = kendalltau(outs_val, gt_val)[0]

        print(f"Valid Loss: {mean_loss:.3f} | Spearman: {corr_spearman:.3f} | Pearson: {corr_pearsonr:.3f} | Kendall: {corr_kendall:.3f}")

        # Validation

        # model.eval()
        # val_loss = []
        # outs_val = []
        # gt_val = []
        # with torch.no_grad():
        #     for data in tqdm.tqdm(val_loader):
        #         with torch.cuda.amp.autocast():
        #             reference = data["reference"].to(DEVICE)
        #             distorted = data["distorted"].to(DEVICE)
        #             labels = data["iou_dist"].to(DEVICE).float()
        #             outputs = model(reference, distorted)
        #             loss = criterion(outputs, labels[:, None])
        #         val_loss.append(loss.item())
        #         outs_val.extend(list(outputs[:, 0].detach().cpu().numpy()))
        #         gt_val.extend(list(labels.detach().cpu().numpy()))
        #
        # mean_loss = np.mean(val_loss)
        # corr_spearman = spearmanr(outs_val, gt_val)[0]
        # corr_pearsonr = pearsonr(outs_val, gt_val)[0]
        # corr_kendall = kendalltau(outs_val, gt_val)[0]
        #
        # print(f"Valid Loss: {mean_loss:.3f} | Spearman: {corr_spearman:.3f} | Pearson: {corr_pearsonr:.3f} | Kendall: {corr_kendall:.3f}")
        #
        # if corr_spearman > best_spearman:
        #     best_spearman = corr_spearman
        #
        #     save_data = {
        #         "model": model,
        #         "optimizer": optimizer
        #     }
        #     torch.save(save_data, checkpoint_path)


def evaluate(checkpoint, val_loader):
    if Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=torch.device(DEVICE))
        model = ckpt["model"]
        print("Model loaded from checkpoint")
    else:
        print("Checkpoint not found, stop evaluation")
        return

    model.eval()
    out_index = []
    val_loss = []
    outs = []
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader):
            with torch.cuda.amp.autocast():
                reference = data["reference"].to(DEVICE)
                distorted = data["distorted"].to(DEVICE)
                labels = data["iou_dist"].to(DEVICE).float()
                indices = data["index"] # shape (b)
                outputs = model(reference, distorted)
            out_index += list(indices.numpy())
            outs_flat = outputs.detach().flatten()
            outs += list(outs_flat.cpu().numpy())
    df = pd.DataFrame({
        "index": out_index,
        "prediction": outs
    })
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    checkpoint_path = "checkpoints/proxy_model.pth"
    is_evaluate = False
    batch_size = 10
    workers = 1

    if is_evaluate:
        # Evaluate model on whole dataset

        coco_base = BaseDataset(csv_path="../datasets/object_results/huawei_objects/yolov5s.csv",
                                data_path="../datasets/huawei_objects_decoded")
        test_dataset = TrainDataset(coco_base, no_split=True)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)

        evaluate(checkpoint_path, test_loader)
    else:
        # Train on 70%, validate on 30%

        coco_base = BaseDataset(csv_path="../datasets/object_results/coco_5k_v3/yolov5s.csv",
                                data_path="../datasets/coco_5k_v3_decoded")
        train_dataset = TrainDataset(coco_base, validation=False)
        val_dataset =   TrainDataset(coco_base, validation=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, drop_last=True, pin_memory=False)
        val_loader =   DataLoader(val_dataset,   batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)

        model = FeatureExtractor("layer4").to(DEVICE)
        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
            model = ckpt["model"]
            print("Model loaded from checkpoint")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion_l1 = nn.L1Loss()

        num_epochs = 1
        train(model, optimizer, criterion_l1, train_loader, val_loader, num_epochs, checkpoint_path)
