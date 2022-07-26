import sys
import argparse
from glob import glob
from pathlib import Path
import time
import copy
import os
import datetime as dt

import numpy as np
import pandas as pd
import cv2
import einops

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr

import torch
import torch.utils.data as data
from torch import nn
import torchvision

import matplotlib.pyplot as plt

import wandb


DEVICE = 'cuda'

# read CSV (duplicate from repo)
IMG_KEY = 'img_basename'
DAT_KEY = 'dataset_name'

def check(table):
    per_img_count = table[[IMG_KEY, DAT_KEY]].groupby(IMG_KEY).count()
    dataset_count = len(table[DAT_KEY].unique())
    if not (per_img_count.values == dataset_count).all():
        problem_images = per_img_count[per_img_count.values != dataset_count].index.values
        print("Warning: in pd.DataFrame index is not complete!")
        print(f"Following {len(problem_images)} image(s) do not exist in every dataset")
        print(problem_images)

def read_results_csv(csv_fname):
    dataframe = pd.read_csv(csv_fname, dtype={IMG_KEY: str, DAT_KEY: str})
    dataframe = dataframe.sort_values([IMG_KEY, DAT_KEY])
    check(dataframe)
    return dataframe

# ********************************* COCO dataset

def get_path(row):
    name, codec = row["img_basename"], row["dataset_name"]
    if "jpeg" in codec or "ref" in codec:
        extension = "jpg"
    else:
        extension = "png"
    return f"../datasets/coco_5k_v3_decoded/{codec}/{name}.{extension}"

class cocoDataset(data.Dataset):
    """
    Dataset returns: (str: image_name, str: degradation, tensor: image, tensor: gt_meaniou)
    GT mean-iou is one number.
    """

    def __init__(self, validation=False):
        self.validation = validation

        # "../datasets/coco_5k_v3_decoded/av1_150/000011.jpg"
        df = read_results_csv("../datasets/coco_5k_results/merged.csv")
        df["path"] = df.apply(get_path, axis=1) # generate image paths from dataframe
        df = df[df["labels"] != 0].reset_index(drop=True) # Remove images with no GT labels

        train_names, val_names = train_test_split(df["img_basename"].unique(), test_size=0.3, random_state=1543)
        self.train_data = df[df["img_basename"].isin(train_names)].reset_index(drop=True).copy()
        self.val_data   = df[df["img_basename"].isin(val_names)  ].reset_index(drop=True).copy()

        if not validation:
            self.data = self.train_data
        else:
            self.data = self.val_data

    def __getitem__(self, index):
        index = index % self.data.shape[0]
        row = self.data.iloc[index]
        image = cv2.imread(row["path"])
        image = image[:, :, ::-1].astype(np.float32) / 255

        input_size = 224
        image = cv2.resize(image, (input_size, input_size))

        # get mean-iou for image from dataframe
        gt = row["yolov5s"]

        return torch.from_numpy(image), torch.tensor( [gt] )

    def __len__(self):
        return self.data.shape[0]


def test_dataset():
    train_dataset = cocoDataset()
    # f, i = train_dataset[0]
    # print(f.shape, i.shape)
    print("Length of dataset:", len(train_dataset))

test_dataset()


class ProxyModel(nn.Module):
    """
    Input tensor must have shape (b 3 224 224)
    """

    def __init__(self, vgg16_model):
        super().__init__()

        self.vgg16 = vgg16_model
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=1, bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=4096, out_features=4096, bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=4096, out_features=1, bias=True),
        )

    def __call__(self, x):
        x = self.vgg16(x).detach()  # stop gradient?
        x = self.head(x)
        return x

    def _conv(self, in_channels, out_channels, kernel_size, relu=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        if relu:
            block.add_module("relu", nn.ReLU())

        return block


PRINT_FREQ = 1


def train(model, train_dl, val_dl, optimizer, visualize_list, train_steps, print_freq=10):
    """
    Main training loop. Load training data, run model forward and backward pass,
    print training status, run validation.
    """
    start = time.time()
    model.to(DEVICE)

    print("Training steps:", train_steps)
    wandb.init(project="my-test-project", config={"speed": 100500})

    step = 0
    ema_loss = 0.0
    start_time = time.time()

    while True:
        model.train(True)  # set training mode
        dataloader = train_dl

        # iterate over data
        for image, target in dataloader:
            if step >= train_steps:
                break

            step += 1

            # moving tensors to GPU
            # tensor shape b h w c -> b c h w
            image = image.to(DEVICE).permute(0, 3, 1, 2).float().contiguous()  # contiguous speeds up 2-4x
            target = target.to(DEVICE)

            # forward pass
            optimizer.zero_grad()

            out = model(image)  # shape=(b c 1 1)

            loss = torch.mean((out - target) ** 2)
            # print("Output shape", out.shape)
            # print("Target", target)

            # backward pass
            loss.backward()
            optimizer.step()

            show_loss = loss.detach().cpu().numpy()
            #             ema_loss = show_loss * (2/(1+step)) + ema_loss * (1-2/(1+step))
            #             metrics = {
            #                 'cross_entropy_loss': show_loss
            #             }
            #             logger.push(metrics)

            if step % print_freq == print_freq - 1:
                print("step: {}, MSE loss: {:.2f} step time: {:.2f}".format(step,
                                                                            show_loss,
                                                                            (time.time() - start_time) / step),
                      flush=True)

                wandb.log({
                    "step": step,
                    "MSE loss": show_loss,
                    "step time": (time.time() - start_time) / step,
                    })

                # Save model

                model_path = Path("checkpoints/proxy_model.pth")

                checkpoint = {
                    'model': model,
                    'optimizer': optimizer,
                    'step': step
                }
                torch.save(checkpoint, model_path)

            #             if step % VAL_FREQ == VAL_FREQ - 1:
            #                 val_error = calc_validation_score()
            #                 metrics = {
            #                     'val_error_percent': val_error
            #                 }
            #                 logger.write_dict(metrics)
            #                 print("validation error: {:.2f}".format(val_error))

        # ************************************** Validation
        model.train(False)  # set training mode
        dataloader = val_dl
        val_step = 0
        val_loss = 0.0

        out_list = []
        target_list = []

        for image, target in dataloader:
            val_step += image.shape[0]
            # if val_step > train_steps:
            #     break

            # moving tensors to GPU
            # tensor shape b h w c -> b c h w
            image = image.to(DEVICE).permute(0, 3, 1, 2).float().contiguous()  # contiguous speeds up 2-4x
            target = target.to(DEVICE)

            out = model(image)  # shape=(b c 1 1)
            loss = torch.sum((out - target) ** 2)

            out_list += list( out.detach().cpu().numpy().flatten() )
            target_list += list( target.detach().cpu().numpy().flatten() )

            # print("Output shape", out.shape)
            # print("Target", target)

            show_loss = loss.detach().cpu().numpy()
            val_loss += show_loss

        x_data = np.array(out_list)
        y_data = np.array(target_list)
        corrp = pearsonr(x_data, y_data)[0]
        corrs = spearmanr(x_data, y_data)[0]

        print("step: {}, validation MSE: {:.2f}, pearson: {:.2f}, spearman: {:.2f}".format(step,
                                                        val_loss / val_step,
                                                        corrp,
                                                        corrs
                                                        ),
              flush=True)

        wandb.log({
            "step": step,
            "validation": val_loss / val_step,
            "pearson": corrp,
            "spearman": corrs,
            })

        if step >= train_steps:
            return

    # return train_loss, valid_loss, disp_vis

def main():
    vgg16_model = torchvision.models.vgg16(pretrained=True)

    # Freeze training for all layers
    for param in vgg16_model.features.parameters():
        param.requires_grad = False

    modules = list(vgg16_model.children())[:-1]  # remove linear block
    vgg16_model = nn.Sequential(*modules)

    train_dataset = cocoDataset()
    val_dataset = cocoDataset(validation=True)
    model = ProxyModel(vgg16_model)

    # Load model weights
    model_path = Path("checkpoints/proxy_model.pth")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        step = checkpoint['step']
        print('Continue from', step, 'step')
    else:
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=0.001)

    experiment = dt.datetime.now().strftime("%H%M%S")
    # logger = Logger(path="runs/logbook-" + experiment)
    visualize_list = []  # [(disparity image, steps performed), ...]
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    train_loader = data.DataLoader(train_dataset, batch_size=200,
                                   pin_memory=False, shuffle=True, num_workers=10, drop_last=True)
    val_loader   = data.DataLoader(val_dataset, batch_size=200,
                                   pin_memory=False, shuffle=False, num_workers=10, drop_last=True)
    train(model, train_loader, val_loader, optimizer, visualize_list, 1)

if __name__ == "__main__":
    main()