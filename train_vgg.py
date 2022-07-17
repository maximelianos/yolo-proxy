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
        # gt = self.df[(self.df["img_basename"] == name) & (self.df["dataset_name"] == codec)]["yolov5s"]
        # gt = gt[0]

        return name, codec, torch.from_numpy(image), torch.tensor( [1] )

    def __len__(self):
        return len(self.image_path)


def test_dataset():
    train_dataset = cocoDataset()
    # f, i = train_dataset[0]
    # print(f.shape, i.shape)
    print("Length of dataset:", len(train_dataset))
    print(train_dataset.image_path[1000])

test_dataset()