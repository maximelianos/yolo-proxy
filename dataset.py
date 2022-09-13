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

class BaseDataset:
    def __init__(self, csv_path, data_path):
        # example: "../datasets/coco_5k_v3_decoded/av1_150/000011.jpg"
        df = read_results_csv(csv_path)
        df["path"] = df.apply(partial(get_path, prepend=data_path), axis=1)  # add column with paths to images
        df["index"] = df.index  # add index to record output of model
        df = df.set_index(["img_basename", "dataset_name"], drop=False)  # fast (?) access to rows

        # train-test split
        train_names, val_names = train_test_split(df["img_basename"].unique(), test_size=0.3, random_state=1543)
        self.train_data = df[df["img_basename"].isin(train_names)].copy()
        self.val_data = df[df["img_basename"].isin(val_names)].copy()
        self.data = df

        # list of possible codec + quality variants, exclude original: av1_150, h264_35, ...
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

        self.image_list = self.data[["img_basename", "dataset_name"]].drop_duplicates()

    # getitem-based dataset (instead of iterable dataset) allows shuffling
    def __getitem__(self, index):
        input_size = 224

        index = index % len(self) - 1

        while True:
            index += 1

            # select image name + degradation
            name_dist = self.image_list.iloc[index]
            name, codec = name_dist["img_basename"], name_dist["dataset_name"]

            # select all objects in image
            df_dist = self.data.loc[name, codec]

            row_dist = df_dist.iloc[0]
            row_ref = self.data.loc[name, "ref"].iloc[0]  # select (name, codec) using multiindex + select first row

            # print("Reference", row_ref["path"])
            # print("Distorted", row_dist["path"])

            image_ref = cv2.imread(row_ref["path"])
            image_ref = image_ref[:, :, ::-1].astype(np.float32) / 255

            image_dist = cv2.imread(row_dist["path"])
            image_dist = image_dist[:, :, ::-1].astype(np.float32) / 255

            # crop random rectangle with fixed height and width
            h, w, c = image_ref.shape
            boxw = 1920
            boxh = 1080
            padx = (max(w, boxw) - w) // 2 + 1  # if boxw > w, add padding
            pady = (max(h, boxh) - h) // 2 + 1
            xmin = torch.randint(0, w - min(w, boxw) + 1, size=(1,))  # if boxw > w, choose 0
            ymin = torch.randint(0, h - min(h, boxh) + 1, size=(1,))
            xmax = xmin + boxw
            ymax = ymin + boxh

            pad = np.pad(image_ref, ((pady, pady), (padx, padx), (0, 0)), "mean")
            crop_ref = pad[ymin:ymax, xmin:xmax]
            pad = np.pad(image_dist, ((pady, pady), (padx, padx), (0, 0)), "mean")
            crop_dist = pad[ymin:ymax, xmin:xmax]

            # select bboxes inside random rect
            MAX_BBOXES = 50
            bboxes = []  # [ [x1 y1 x2 y2], ... ]
            ious = []  # [ i1 i2 ... ]
            for _index, b in df_dist.iterrows():
                b.xmin += padx
                b.xmax += padx
                b.ymin += pady
                b.ymax += pady
                if (xmin <= b.xmin and b.xmax < xmax and
                        ymin <= b.ymin and b.ymax < ymax
                ):
                    bboxes.append([b.xmin - xmin, b.ymin - ymin, b.xmax - xmin, b.ymax - ymin])
                    ious.append(b.iou)

            t_bboxes = torch.tensor(bboxes)  # (k, 4)
            if len(t_bboxes.shape) < 2:
                continue
            k, _coors = t_bboxes.shape
            t_bboxes = nn.functional.pad(t_bboxes, (0, 0, 0, MAX_BBOXES - k))

            t_ious = torch.tensor(ious)
            t_ious = nn.functional.pad(t_ious, (0, MAX_BBOXES - k))

            return {
                "reference": torch.tensor(crop_ref).permute(2, 0, 1).float(),  # (c, h, w)
                "distorted": torch.tensor(crop_dist).permute(2, 0, 1).float(),  # (c, h, w)
                "bbox": t_bboxes.float(),  # (k, 4)
                "iou_dist": t_ious.float(),  # (k)
                "index": index
            }

    def __len__(self):
        return self.image_list.shape[0]
