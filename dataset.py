import time
import os
from functools import partial

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torchvision


# ********************************* read CSV (duplicate from repo)

IMG_KEY = "img_basename"
DAT_KEY = "dataset_name"

def read_results_csv(csv_fname):
    dataframe = pd.read_csv(csv_fname, dtype={IMG_KEY: str, DAT_KEY: str})
    dataframe = dataframe.sort_values([IMG_KEY, DAT_KEY])
    return dataframe


# ********************************* COCO dataset

def get_path(row, prepend):
    # Utility function, construct path to images
    name, codec = row["img_basename"], row["dataset_name"]
    if "huawei" in prepend:
        if "jpeg" in codec:
            extension = "jpg"
        else:
            extension = "png"
    else: # coco
        if "jpeg" in codec or "ref" in codec:
            extension = "jpg"
        else:
            extension = "png"
    return f"{prepend}/{codec}/{name}.{extension}"

def split_names(csv_path, names, test_size=0.3):
    # TODO: Implement different logic for datasets
    if 'huawei' in csv_path:
        # shuffle videos
        names["videoname"] = names[IMG_KEY].apply(lambda filename: filename.split(".mp4")[0])
        train_videos, val_videos = train_test_split(names["videoname"].unique(), test_size=test_size, random_state=1543, )
        # print(train_videos)
        # print(val_videos)
        return (names[names["videoname"].isin(train_videos)][IMG_KEY], names[names["videoname"].isin(val_videos)][IMG_KEY])
    else:
        return train_test_split(names[IMG_KEY].unique(), test_size=test_size, random_state=1543, )

class BaseDataset:
    def __init__(self, csv_path, data_path):
        # example: "../datasets/coco_5k_v3_decoded/av1_150/000011.jpg"
        df = read_results_csv(csv_path)
        df["path"] = df.apply(partial(get_path, prepend=data_path), axis=1)  # add column with paths to images
        df["index"] = df.index  # add index to record output of model
        df = df.set_index(["img_basename", "dataset_name", "object_id"], drop=False)  # fast (?) access to rows

        # train-test split
        name_df = pd.DataFrame(df["img_basename"].unique(), columns=[IMG_KEY])
        train_names, val_names = split_names(csv_path, name_df)
        self.train_data = df[df["img_basename"].isin(train_names)].copy()
        self.val_data = df[df["img_basename"].isin(val_names)].copy()
        self.data = df

        # list of possible codec + quality variants, excluding reference: av1_150, h264_35, ...
        dists_set = set(df["dataset_name"].unique())
        self.dists = list(dists_set - {"ref"})


class TrainDataset(torch.utils.data.Dataset):
    """
    Return reference and distorted object images. Used for training, validation and testing.
    Dataset returns:
    {
        reference: b c h w
        distorted: b c h w
        iou_dist: float, intersection over union for distorted object
        index: int, index in dataset
    }
    """

    def __init__(self, base_dataset, validation=False, no_split=False):
        self.validation = validation
        self.dists = base_dataset.dists
        if no_split:
            self.data = base_dataset.data
        elif validation:
            self.data = base_dataset.val_data
        else:
            self.data = base_dataset.train_data

    # getitem-based dataset (instead of iterable dataset) allows shuffling
    def __getitem__(self, index):
        input_size = 224

        index = index % len(self)

        # select degradation
        row_dist = self.data.iloc[index]
        name, codec, objid = row_dist["img_basename"], row_dist["dataset_name"], row_dist["object_id"]
        xmin, ymin, xmax, ymax = map(int, [row_dist.xmin, row_dist.ymin, row_dist.xmax, row_dist.ymax])
        row_ref = self.data.loc[name, "ref", objid]

        # print("Reference", row_ref["path"])
        # print("Distorted", row_dist["path"])

        image_ref = cv2.imread(row_ref["path"])
        image_ref = image_ref[:, :, ::-1].astype(np.float32) / 255
        image_ref = image_ref[ymin:ymax, xmin:xmax]
        image_ref = cv2.resize(image_ref, (input_size, input_size))

        image_dist = cv2.imread(row_dist["path"])
        image_dist = image_dist[:, :, ::-1].astype(np.float32) / 255
        image_dist = image_dist[ymin:ymax, xmin:xmax]
        image_dist = cv2.resize(image_dist, (input_size, input_size))

        return {
            "reference": torch.tensor(image_ref).permute(2, 0, 1),
            "distorted": torch.tensor(image_dist).permute(2, 0, 1),
            "iou_dist": row_dist["iou"],
            "index": row_dist["index"]
        }

    def __len__(self):
        return self.data.shape[0]
