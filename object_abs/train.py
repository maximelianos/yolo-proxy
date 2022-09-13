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
        self.fc = nn.Linear(512 * 1, 1)

    def forward(self, ref, dist):
        b, c, h, w = ref.shape
        # a = self.backbone(ref)  # (b, c, h, w) -> (b, 512)
        b = self.backbone(dist)
        # concat = torch.cat((a, b, a-b), dim=1)  # (b, 512*3)
        x = self.fc(b)  # (b, 1)
        return x


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint_path):
    best_spearman = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        for is_train in [True, False]:
            if is_train:
                print("== Train ==")
                model.train()
            else:
                print("== Validation ==")
                model.eval()

            step_loss = []
            step_outs = []
            step_gt = []
            step = 1

            loader = train_loader if is_train else val_loader
            with torch.set_grad_enabled(is_train):
                for data in tqdm.tqdm(loader):
                    if is_train:
                        optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        reference = data["reference"].to(DEVICE)  # shape (b c h w)
                        distorted = data["distorted"].to(DEVICE)  # shape (b c h w)
                        labels = data["iou_dist"].to(DEVICE).float()  # shape (b)
                        outputs = model(reference, distorted)  # shape (b 1)
                        loss = criterion(outputs, labels[:, None])
                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    step_loss.append(loss.item())
                    step_outs.extend(list(outputs[:, 0].detach().cpu().numpy()))
                    step_gt.extend(list(labels.detach().cpu().numpy()))
                    step += 1

                mean_loss = np.mean(step_loss)
                corr_spearman =  spearmanr(step_outs, step_gt)[0]
                corr_pearsonr =   pearsonr(step_outs, step_gt)[0]
                corr_kendall  = kendalltau(step_outs, step_gt)[0]

                print(f"Loss: {mean_loss:.3f} | Spearman: {corr_spearman:.3f} | Pearson: {corr_pearsonr:.3f} | Kendall: {corr_kendall:.3f}", flush=True)

                if corr_spearman > best_spearman:
                    best_spearman = corr_spearman

                    save_data = {
                        "model": model,
                        "optimizer": optimizer
                    }
                    torch.save(save_data, checkpoint_path)


def evaluate(checkpoint, val_loader):
    if Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=torch.device(DEVICE))
        model = ckpt["model"]
        print("Model loaded from checkpoint")
    else:
        print("Checkpoint not found, stop evaluation")
        return

    model.eval()
    indices = []
    step_outs = []
    step_gt = []
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader):
            with torch.cuda.amp.autocast():
                reference = data["reference"].to(DEVICE)
                distorted = data["distorted"].to(DEVICE)
                labels = data["iou_dist"].to(DEVICE).float()
                outputs = model(reference, distorted)
            indices += list(data["index"].numpy()) # shape (b)
            outs_flat = outputs.detach().flatten()
            step_outs += list(outs_flat.cpu().numpy())
            step_gt += list(labels.cpu().numpy())
    corr_spearman =  spearmanr(step_outs, step_gt)[0]
    corr_pearsonr =   pearsonr(step_outs, step_gt)[0]
    corr_kendall  = kendalltau(step_outs, step_gt)[0]
    print(f"Spearman: {corr_spearman:.3f} | Pearson: {corr_pearsonr:.3f} | Kendall: {corr_kendall:.3f}", flush=True)
    df = pd.DataFrame({
        "index": indices,
        "prediction": step_outs
    })
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    checkpoint_path = "checkpoints/proxy_model.pth"
    is_evaluate = False
    batch_size = 128
    workers = 32

    if is_evaluate:
        # Evaluate model on whole dataset

        coco_base = BaseDataset(csv_path="../datasets/object_results/huawei_objects/yolov5s.csv",
                                data_path="../datasets/huawei_objects_decoded")
        # coco_base = BaseDataset(csv_path="../datasets/object_results/coco_5k_v3/yolov5s.csv",
        #                         data_path="../datasets/coco_5k_v3_decoded")
        test_dataset = TrainDataset(coco_base, validation=True)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)

        evaluate(checkpoint_path, test_loader)
    else:
        # Train on 70%, validate on 30%

        coco_base = BaseDataset(csv_path="../datasets/object_results/coco_5k_v3/yolov5s.csv",
                                data_path="../datasets/coco_5k_v3_decoded")
        # coco_base = BaseDataset(csv_path="../datasets/object_results/huawei_objects/yolov5s.csv",
        #                         data_path="../datasets/huawei_objects_decoded")
        train_dataset = TrainDataset(coco_base, validation=False)
        val_dataset =   TrainDataset(coco_base, validation=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, drop_last=True, pin_memory=False)
        val_loader =   DataLoader(val_dataset,   batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)

        model = Net().to(DEVICE)
        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
            model = ckpt["model"]
            print("Model loaded from checkpoint")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion_l1 = nn.L1Loss()

        num_epochs = 20
        train(model, optimizer, criterion_l1, train_loader, val_loader, num_epochs, checkpoint_path)
