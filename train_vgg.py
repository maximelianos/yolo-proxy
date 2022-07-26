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

import torch
import torch.utils.data as data
from torch import nn
import torchvision

import matplotlib.pyplot as plt

import wandb
#wandb.init(project="my-test-project", config={"speed": 100500})

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


# load dataframe (duplicate from repo)

# general_metrics = []
# RANDOM_STATE = 1234
#
#
# class Context:
#     # Base data
#     metric_data = None  # data from csv with filtered 'labels' and 'ref'
#     TRAIN_METRICS = None  # training columns
#     TARGET_METRIC = None  # target column
#     DATASET_NAME = None
#
#     # Split data (with all columns)
#     train_data = None
#     val_data = None
#
#     # Data for model training (only relevant columns)
#     X_train = None
#     X_train_scaled = None
#     Y_train = None
#
#     X_val = None
#     X_val_scaled = None
#     Y_val = None
#
#     # sklearn classes
#     scaler = None
#     model_svr = None
#
#
# def load_dataset(ctx: Context, csv_path: str, target_metric: str):
#     print(f"=== Loading dataset from {csv_path} ===")
#     df = read_results_csv(csv_path)
#     df = df.fillna(df.mean(numeric_only=True))
#     ctx.DATASET_NAME = Path(csv_path).parent.name
#     print("Unique image filenames:", len(df[IMG_KEY].unique()))
#
#     columns_without_superfast = {
#         c: c.replace(' superfast-', ' fast-')
#         for c in df.columns
#     }
#     df.rename(columns=columns_without_superfast, inplace=True)  # metric naming in VQMT 13->14 versions
#
#     if 'huawei' in ctx.DATASET_NAME:  # Unstable naming of vmaf in VQMT
#         df.rename(columns={"netflix vmaf_vmaf061-y": "netflix vmaf_vmaf061_float-y"}, inplace=True)
#
#     # Select only relevant images
#     if 'labels' in df.columns:
#         df = df[df['labels'] > 0]  # use images only with some objects on them
#
#     # Remove ref data from analyzing
#     # df = df[df[DAT_KEY] != 'ref']
#
#     ctx.TRAIN_METRICS = sorted([c for c in df.columns if c in general_metrics])
#     print('Used metrics:', ctx.TRAIN_METRICS)
#     ctx.TARGET_METRIC = target_metric
#     ctx.metric_data = df.copy()  # Warning: columns are not sorted
#
#
# def split_names(ctx, names, test_size):
#     # TODO: Implement different logic for datasets
#     if 'huawei' in ctx.DATASET_NAME:
#         # shuffle videos
#         names["videoname"] = names[IMG_KEY].apply(lambda filename: filename.split(".mp4")[0])
#         train_videos, val_videos = train_test_split(names["videoname"].unique(), test_size=test_size, random_state=RANDOM_STATE, )
#         return (names[names["videoname"].isin(train_videos)][IMG_KEY], names[names["videoname"].isin(val_videos)][IMG_KEY])
#     else:
#         return train_test_split(names[IMG_KEY].unique(), test_size=test_size, random_state=RANDOM_STATE, )
#
#
# def split_dataset(ctx: Context, test_size=0.3, train_images=1000, test_only=False):
#     # Split data, group by image names
#     train_names, val_names = split_names(ctx, ctx.metric_data.loc[:, [IMG_KEY]].copy(), test_size=test_size)  # get column as df, not series
#     if train_images is not None:
#         train_names = train_names[:train_images]
#
#     ctx.train_data = ctx.metric_data[ctx.metric_data[IMG_KEY].isin(train_names)].reset_index(drop=True).copy()
#     ctx.val_data = ctx.metric_data[ctx.metric_data[IMG_KEY].isin(val_names)].reset_index(drop=True).copy()
#     if test_only:
#         ctx.train_data = ctx.metric_data.copy()
#         ctx.val_data = ctx.metric_data.copy()
#
#     ctx.train_data = shuffle(ctx.train_data, random_state=RANDOM_STATE)
#     ctx.val_data = shuffle(ctx.val_data, random_state=RANDOM_STATE)
#     ctx.X_train = ctx.train_data[ctx.TRAIN_METRICS]
#     ctx.Y_train = ctx.train_data[ctx.TARGET_METRIC]
#
#     ctx.X_val = ctx.val_data[ctx.TRAIN_METRICS]
#     ctx.Y_val = ctx.val_data[ctx.TARGET_METRIC]
#
#     # Scale data
#     if not ctx.scaler:
#         ctx.scaler = StandardScaler()
#         ctx.scaler.fit(ctx.X_train)
#
#     ctx.X_train_scaled = ctx.scaler.transform(ctx.X_train)
#     ctx.X_val_scaled = ctx.scaler.transform(ctx.X_val)

# ********************************* COCO dataset

def get_path(row):
    name, codec = row["img_basename"], row["dataset_name"]
    if "jpeg" in codec:
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
        if not Path(row["path"]).exists():
            print("Path not found:", row["path"])
        else:
            print("Path ok:", row["path"])
        image = cv2.imread(row["path"])
        if image is None:
            print("Image not loaded:", row["path"])
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
            nn.Linear(in_features=25088, out_features=1, bias=True)
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


def train(model, train_dl, val_dl, optimizer, visualize_list, train_steps, print_freq=1):
    """
    Main training loop. Load training data, run model forward and backward pass,
    print training status, run validation.
    """
    start = time.time()
    model.to(DEVICE)

    print("Training steps:", train_steps)

    while True:
        model.train(True)  # set training mode
        dataloader = train_dl
        step = 0
        ema_loss = 0.0
        start_time = time.time()

        # iterate over data
        for image, target in dataloader:
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

                # wandb.log({
                #     "step": step,
                #     "MSE loss": show_loss,
                #     "step time": (time.time() - start_time) / step,
                #     })

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

            if step > train_steps:
                return

        # ************************************** Validation
        model.train(False)  # set training mode
        dataloader = val_dl
        val_step = 0
        val_loss = 0.0

        for image, target in dataloader:
            val_step += image.shape[0]

            # moving tensors to GPU
            # tensor shape b h w c -> b c h w
            image = image.to(DEVICE).permute(0, 3, 1, 2).float().contiguous()  # contiguous speeds up 2-4x
            target = target.to(DEVICE)

            out = model(image)  # shape=(b c 1 1)

            loss = torch.sum((out - target) ** 2)
            # print("Output shape", out.shape)
            # print("Target", target)

            show_loss = loss.detach().cpu().numpy()
            val_loss += show_loss

        print("validation:", val_loss / val_step)

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
    train_loader = data.DataLoader(train_dataset, batch_size=1,
                                   pin_memory=False, shuffle=True, num_workers=1, drop_last=True)
    val_loader   = data.DataLoader(val_dataset, batch_size=1,
                                   pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
    train(model, train_loader, val_loader, optimizer, visualize_list, 10)

if __name__ == "__main__":
    main()