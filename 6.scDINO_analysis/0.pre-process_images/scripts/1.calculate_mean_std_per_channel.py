#!/usr/bin/env python
# coding: utf-8

# This notebook calculates the mean and std per channel.
# Having this information is useful to normalize the data for downstream scDINO representation learning.

# In[1]:


import json
import pathlib

import numpy as np
import pandas as pd
import skimage.io
import torch
import tqdm
from tifffile import imread
from torchvision import datasets

# In[2]:


# set the path to the data
data_dir = pathlib.Path("../data/processed_images/sc_crops/").resolve(strict=True)

# output path
output_file_path = pathlib.Path(
    "../data/processed_images/mean_std_normalization/mean_std.txt"
).resolve()
# make sure the output directory exists
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# get a list of files recursively (.tiff files) specified in the data_dirs
files = list(data_dir.glob("**/*.tiff"))
# get files
image_list = [f for f in files if f.is_file()]
print(f"Found {len(files)} files")


# In[3]:


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = imread(path)
        image = image.astype(float)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        if torch.isnan(tensor).any():
            print("nan in tensor: ", path)
            return None
        else:
            return tensor, idx


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# In[4]:


dataset_total = ReturnIndexDataset(data_dir)
shuffle_dataset = True
random_seed = 0
dataset_size = len(image_list)
indices = list(range(dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)


sampler = torch.utils.data.SubsetRandomSampler(indices)

image_data_loader = torch.utils.data.DataLoader(
    dataset_total,
    sampler=sampler,
    batch_size=int(len(indices) / 10),
    num_workers=0,
    collate_fn=collate_fn,
    drop_last=True,
)


# In[5]:


def batch_mean_and_sd(loader):
    cnt = 0
    picture, _ = next(iter(image_data_loader))
    b, c, h, w = picture.shape
    fst_moment = torch.empty(c)
    snd_moment = torch.empty(c)

    for images, _ in loader:
        b, c, h, w = images.shape
        print(b, c, h, w)
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
    return mean, std


# In[6]:


# run the function to get the per channel mean and std
mean, std = batch_mean_and_sd(image_data_loader)
print("mean and std: \n", mean, std)


# In[7]:


# scale the mean and std to 0-1

# get the image bit depth from skimage
image = imread(image_list[0])
image_max_bit = np.max(image)

if image_max_bit <= 255:
    mean = mean / 255
    std = std / 255
elif image_max_bit <= 65535:
    mean = mean / 65535
    std = std / 65535
else:
    raise ValueError("Image bit depth not supported")

# print the mean and std
print("mean and std: \n", mean, std)

with open(output_file_path, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
