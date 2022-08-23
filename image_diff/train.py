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
from torch.utils import data
from torch import nn
import torchvision
from torchvision.models import resnet18

import tqdm
import matplotlib.pyplot as plt

from dataset import BaseDataset, TrainDataset, TestDataset


DEVICE = "cuda"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)
        
    def forward(self, ref, dist):
        b, s, c, h, w = ref.shape
        a = self.backbone(ref.view((-1, c, h, w)))  # (b*s, c, h, w) -> (b*s, 512)
        b = self.backbone(dist.view((-1, c, h, w)))
        diff = (a - b).view((-1, s, 512))  # (b, s, 512)
        pooled = diff.mean(dim=1)  # (b, 512)
        x = self.fc(pooled)  # (b, 1)
        return x


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint='model.pth'):
    best_spearman = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        step = 0
        for data in tqdm.tqdm(train_loader):
            # if step > 10:
            #     break
            step += 1
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reference = data['reference'].to(DEVICE) # shape (b s c h w)
                distorted = data['distorted'].to(DEVICE) # shape (b s c h w)
                labels = data['delta_target'].to(DEVICE).float() # shape (b)
                outputs = model(reference, distorted) # shape (b 1)
                loss = criterion(outputs, labels[:, None])
                #print("L1 (MSE):", loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = []
        outs_val = []
        gt_val = []
        with torch.no_grad():
            step = 0
            for data in tqdm.tqdm(val_loader):
                # if step > 10:
                #     break
                step += 1
                with torch.cuda.amp.autocast():
                    reference = data['reference'].to(DEVICE)
                    distorted = data['distorted'].to(DEVICE)
                    labels = data['delta_target'].to(DEVICE).float()
                    outputs = model(reference, distorted)
                    loss = criterion(outputs, labels[:, None])
                val_loss.append(loss.item())
                outs_val.extend(list(outputs[:, 0].detach().cpu().numpy()))
                gt_val.extend(list(labels.detach().cpu().numpy()))

        mean_loss = np.mean(val_loss)
        corr_spearman = spearmanr(outs_val, gt_val)[0]
        corr_pearsonr = pearsonr(outs_val, gt_val)[0]
        corr_kendall = kendalltau(outs_val, gt_val)[0]

        print(f"Valid Loss: {mean_loss:.3f} | Spearman: {corr_spearman:.3f} | Pearson: {corr_pearsonr:.3f} | Kendall: {corr_kendall:.3f}")

        if corr_spearman > best_spearman:
            best_spearman = corr_spearman

            save_data = {
                'model': model,
                'optimizer': optimizer
            }
            torch.save(save_data, checkpoint)


def evaluate(checkpoint, val_loader):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        checkpoint = torch.load(checkpoint, map_location=torch.device(DEVICE))
        model = checkpoint["model"]
    else:
        print("Checkpoint not found, stop evaluation")
        return

    model.eval()
    out_index = []
    val_loss = []
    outs_val = []
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader):
            with torch.cuda.amp.autocast():
                reference = data['reference'].to(DEVICE) # shape (b s c h w)
                distorted = data['distorted'].to(DEVICE)
                labels = data['delta_target'].to(DEVICE).float()
                indices = data["index"] # shape (b s)
                outputs = model(reference, distorted)
            b, s, c, h, w = reference.shape
            out_index += list(indices.numpy().flatten())
            outs_dup = outputs.detach().expand(-1, s).flatten()
            outs_val += list(outs_dup.cpu().numpy())
    df = pd.DataFrame({
        "index": out_index,
        "delta": outs_val
    })
    df.to_csv("deltas.csv", index=False)


if __name__ == "__main__":
    checkpoint_path = "checkpoints/proxy_model.pth"
    is_evaluate = True

    batch_size = 16
    workers = 32
    batch_s = 10

    coco_base = BaseDataset(csv_path="../datasets/coco_5k_results/merged.csv",
        data_path="../datasets/coco_5k_v3_decoded")
    train_dataset = TrainDataset(coco_base, validation=False, batch=batch_s)
    val_dataset = TrainDataset(coco_base, validation=True, batch=batch_s)
    test_dataset = TestDataset(coco_base, batch=batch_s)

    criterion_l1 = nn.L1Loss()
    model = Net().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    if is_evaluate:
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
            num_workers=workers, drop_last=False, pin_memory=False)

        evaluate(checkpoint_path, test_loader)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
            num_workers=workers, drop_last=True, pin_memory=False)

        train(model, optimizer, criterion_l1, train_loader, val_loader, 200, checkpoint_path)
