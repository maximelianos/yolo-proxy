import sys
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

import torch
import torch.utils.data as data
from torch import nn
import torchvision

import matplotlib.pyplot as plt


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



class cocoDataset(data.Dataset):
    """
    Dataset returns: (str: image_name, str: degradation, tensor: image, tensor: gt_meaniou)
    GT mean-iou is one number.
    """

    def __init__(self):
        dataset_root = "../datasets"
        self.image_path = []  # gt

        pattern = os.path.join(dataset_root, "coco_5k_v3_decoded/*/*")
        self.image_path = sorted(glob(pattern + ".png") + glob(pattern + ".jpg"))

        self.df = read_results_csv(os.path.join(dataset_root, "coco_5k_results/merged.csv"))

    def __getitem__(self, index):
        index = index % len(self.image_path)
        image = cv2.imread(self.image_path[index])[:, :, ::-1].astype(np.float32) / 255

        path = Path(self.image_path[index])
        name = path.stem
        codec = path.parent.name

        input_size = 224
        image = cv2.resize(image, (input_size, input_size))

        # get mean-iou for image from dataframe
        gt = self.df[(self.df["img_basename"] == name) & (self.df["dataset_name"] == codec)]["yolov5s"]
        gt = gt.iloc[0]

        return name, codec, torch.from_numpy(image), torch.tensor( [gt] )

    def __len__(self):
        return len(self.image_path)


def test_dataset():
    train_dataset = cocoDataset()
    # f, i = train_dataset[0]
    # print(f.shape, i.shape)
    print("Length of dataset:", len(train_dataset))
    print(train_dataset.image_path[1000])

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


def train(model, train_dl, optimizer, visualize_list, train_steps, print_freq=1):
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
        for name, codec, image, target in dataloader:
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

    # return train_loss, valid_loss, disp_vis

def main():
    vgg16_model = torchvision.models.vgg16(pretrained=True)

    # Freeze training for all layers
    for param in vgg16_model.features.parameters():
        param.requires_grad = False

    modules = list(vgg16_model.children())[:-1]  # remove linear block
    vgg16_model = nn.Sequential(*modules)

    train_dataset = cocoDataset()
    model = ProxyModel(vgg16_model)

    experiment = dt.datetime.now().strftime("%H%M%S")
    # logger = Logger(path="runs/logbook-" + experiment)
    visualize_list = []  # [(disparity image, steps performed), ...]
    model = ProxyModel(vgg16_model).to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=0.02)
    train_loader = data.DataLoader(train_dataset, batch_size=200,
                                   pin_memory=False, shuffle=True, num_workers=5, drop_last=True)

    train(model, train_loader, optimizer, visualize_list, 200)

if __name__ == "__main__":
    main()