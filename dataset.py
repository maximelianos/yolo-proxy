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
        df["path"] = df.apply(get_path, axis=1) # add column with paths to images
        df = df[df["labels"] != 0].reset_index(drop=True) # Remove images with no GT labels

        # train-test split
        train_names, val_names = train_test_split(df["img_basename"].unique(), test_size=0.3, random_state=1543)
        self.train_data = df[df["img_basename"].isin(train_names)].reset_index(drop=True).copy()
        self.val_data   = df[df["img_basename"].isin(val_names)  ].reset_index(drop=True).copy()

        # list of possible codec + quality variants, exclude original
        dists_set = set(df["dataset_name"].unique())
        self.dists = list(dists_set - {"ref"})

        if not validation:
            self.data = self.train_data
        else:
            self.data = self.val_data

    def __getitem__(self, _):
        # select degradation
        degrad = np.random.choice(self.dists)
        df = self.data[ self.data["dataset_name"] == degrad ]
        #index = index % self.data.shape[0]

        input_size = 224

        list_reference = []
        list_distorted = []
        list_delta_miou = []

        batch = 2
        for index in np.random.randint(df.shape[0], size=batch):
            row_dist = df.iloc[index]
            img_name = row_dist["img_basename"]
            row_ref = self.data[
                (self.data["dataset_name"] == "ref") & (self.data["img_basename"] == img_name)
            ].iloc[0]

            image_ref = cv2.imread(row_ref["path"])
            image_ref = image_ref[:, :, ::-1].astype(np.float32) / 255
            image_ref = cv2.resize(image_ref, (input_size, input_size))

            image_dist = cv2.imread(row_dist["path"])
            image_dist = image_dist[:, :, ::-1].astype(np.float32) / 255
            image_dist = cv2.resize(image_dist, (input_size, input_size))

            miou_diff = row_ref["yolov5s"] - row_dist["yolov5s"]

            list_reference.append(torch.tensor(image_ref).permute(2, 0, 1))
            list_distorted.append(torch.tensor(image_dist).permute(2, 0, 1))
            list_delta_miou.append(miou_diff)

        return  {
            "reference": torch.stack(list_reference),
            "distorted": torch.stack(list_distorted),
            "delta_target": np.mean(list_delta_miou)
        }

    def __len__(self):
        return self.data.shape[0]



# class StochasticDataset(torch.utils.data.IterableDataset):
#
#     def __init__(self, size, num_samples, celeba_distortions_path, emb_base,
#                  emb_distortions, map_base_name_to_ref, file_names, is_val=False, transform=None):
#         self.celeba_distortions_path = celeba_distortions_path
#         self.emb_base = emb_base
#         self.emb_distortions = emb_distortions
#         self.distortions_folders = sorted(os.listdir(celeba_distortions_path))
#         self.file_names = sorted(file_names)
#         self.transform = transform
#         self.map_base_name_to_ref = map_base_name_to_ref
#         self.num_samples = num_samples
#         self.dist = sorted(os.listdir(self.celeba_distortions_path))
#         self.size = size
#         self.is_val = is_val
#
#     def __len__(self):
#         return self.size
#
#     def __iter__(self):
#
#         for i in range(self.size):
#
#             if self.is_val:
#                 np.random.seed(i)
#             else:
#                 np.random.seed(int(time.time()))
#
#             list_distorted = []
#             list_reference = []
#             list_deltas = []
#
#             distortion_name = np.random.choice(self.dist)
#
#             for num_tryes in range(self.num_samples):
#
#                 base_name = np.random.choice(self.file_names)
#
#                 ref_name = self.map_base_name_to_ref[base_name]
#
#                 if 'jpeg' in distortion_name:
#                     img_distorted = os.path.join(self.celeba_distortions_path, distortion_name, ref_name + '.jpg')
#                 else:
#                     img_distorted = os.path.join(self.celeba_distortions_path, distortion_name, ref_name + '.png')
#
#                 img_reference = os.path.join(self.celeba_distortions_path, 'ref', ref_name + '.png')
#
#                 distorted = self.transform(Image.open(img_distorted).convert('RGB'))
#                 reference = self.transform(Image.open(img_reference).convert('RGB'))
#
#                 if 'jpeg' in distortion_name:
#                     emb_distorted = self.emb_distortions[distortion_name][ref_name + '.jpg']
#                 else:
#                     emb_distorted = self.emb_distortions[distortion_name][ref_name + '.png']
#
#                 emb_ref = self.emb_distortions['ref'][ref_name + '.png']
#                 emb_in_base = self.emb_base[base_name + '.png']
#
#                 a = sklearn.metrics.pairwise.cosine_similarity([emb_ref], [emb_in_base])[0][0]
#                 b = sklearn.metrics.pairwise.cosine_similarity([emb_distorted], [emb_in_base])[0][0]
#
#                 list_distorted.append(distorted)
#                 list_reference.append(reference)
#                 list_deltas.append(a-b)
#
#             yield {
#                     'distorted':torch.stack(list_distorted),
#                     'reference':torch.stack(list_reference),
#                     'delta_with_hidden_target': np.mean(list_deltas)
#                   }