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

from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict

import matplotlib.pyplot as plt


DEVICE = 'cuda'


class cocoDataset(data.Dataset):
    def __init__(self):
        self.image_list = []  # gt

        dataset_root = "data"

        self.image_list = sorted(glob(dataset_root + "/*.jpg"))

    def __getitem__(self, index):
        index = index % len(self.image_list)
        image = cv2.imread(self.image_list[index])[:, :, ::-1].astype(np.float32) /  255

        return image

    def __len__(self):
        return len(self.image_list)


def test_dataset():
    train_dataset = cocoDataset()
    # f, i = train_dataset[0]
    # print(f.shape, i.shape)
    print(train_dataset.image_list[1000])
    print("Length of dataset:", len(train_dataset))
    print(f"images: {len(train_dataset.image_list)}")

test_dataset()