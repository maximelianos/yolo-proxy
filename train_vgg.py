import sys
from glob import glob
from pathlib import Path
import time
import copy
import os
import datetime as dt

import numpy as np
import cv2
import einops

import torch
import torch.utils.data as data
from torch import nn

import matplotlib.pyplot as plt


DEVICE = 'cuda'


class cocoDataset(data.Dataset):
    def __init__(self):
        dataset_root = "../datasets"
        self.image_path = []  # gt

        pattern = os.path.join(dataset_root, "coco_5k_v3_decoded/*/*")
        self.image_path = sorted(glob(pattern + ".png"), glob(pattern + ".jpg"))

    def __getitem__(self, index):
        index = index % len(self.image_path)
        image = cv2.imread(self.image_path[index])[:, :, ::-1].astype(np.float32) / 255

        input_size = 224
        image = cv2.resize(image, (input_size, input_size))

        return self.image_path[index], torch.from_numpy(image), torch.tensor( [np.mean(image)] )

    def __len__(self):
        return len(self.image_path)


def test_dataset():
    train_dataset = cocoDataset()
    # f, i = train_dataset[0]
    # print(f.shape, i.shape)
    print("Length of dataset:", len(train_dataset))
    print(train_dataset.image_path[1000])

test_dataset()