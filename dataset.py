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

def get_path(row, prepend):
    name, codec = row["img_basename"], row["dataset_name"]
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
        df = df[df["labels"] != 0].reset_index(drop=True)  # Remove images with no GT labels
        df["index"] = df.index  # add index to record output of model
        df = df.set_index(["img_basename", "dataset_name"], drop=False)  # fast (?) access to rows

        # train-test split
        train_names, val_names = train_test_split(df["img_basename"].unique(), test_size=0.3, random_state=1543)
        self.train_data = df[df["img_basename"].isin(train_names)].copy()
        self.val_data = df[df["img_basename"].isin(val_names)].copy()

        # list of possible codec + quality variants, exclude original
        dists_set = set(df["dataset_name"].unique())
        self.dists = list(dists_set - {"ref"})


class TrainDataset(torch.utils.data.Dataset):
    """
    Return random elements of dataset. Used for training and validation.
    Dataset returns:
    {
        reference: s c h w
        distorted: s c h w
        delta_target: float - mean delta across s ref-dist pairs
    }
    """

    def __init__(self, base_dataset, validation=False):
        self.validation = validation
        self.dists = base_dataset.dists
        if validation:
            self.data = base_dataset.val_data
        else:
            self.data = base_dataset.train_data

    def __getitem__(self, _):
        input_size = 224

        # select degradation
        degrad = np.random.choice(self.dists)
        df = self.data[ self.data["dataset_name"] == degrad ]

        list_reference = []
        list_distorted = []
        list_delta_miou = []

        batch = 10
        # load <batch> pairs of reference and distorted images
        for index in np.random.randint(df.shape[0], size=batch):
            row_dist = df.iloc[index]
            img_name = row_dist["img_basename"]
            row_ref = self.data.loc[img_name, "ref"]

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


class TestDataset(torch.utils.data.IterableDataset):
    """
    Sequential access. Used for validation and inference.
    Dataset returns:
    {
        reference: s c h w
        distorted: s c h w
        delta_target: float - mean delta across s ref-dist pairs
        index: s - for saving model output
    }
    """

    def __init__(self, base_dataset):
        self.data = base_dataset.val_data
        self.dists = base_dataset.dists

    def __iter__(self):
        input_size = 224

        item_idx = 0

        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id

        # select degradation
        for degrad in self.dists:
            df = self.data[self.data["dataset_name"] == degrad]
            # shuffle (optional?)
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)

            batch = 10
            # select start index of batch
            for i in range(0, df.shape[0], batch):
                item_idx += 1
                if item_idx % worker_total_num != worker_id:
                    # skipping needed for parallel data loading
                    continue

                list_reference = []
                list_distorted = []
                list_delta_miou = []
                list_index = []  # remember row index at inference

                # load batch
                for j in range(batch):
                    index = (i + j) % df.shape[0]

                    row_dist = df.iloc[index]
                    img_name = row_dist["img_basename"]
                    row_ref = self.data.loc[img_name, "ref"]

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
                    list_index.append(row_dist["index"])

                yield {
                    "reference": torch.stack(list_reference),
                    "distorted": torch.stack(list_distorted),
                    "delta_target": np.mean(list_delta_miou),
                    "index": torch.tensor(list_index)
                }

    def __len__(self):
        return self.data.shape[0]
