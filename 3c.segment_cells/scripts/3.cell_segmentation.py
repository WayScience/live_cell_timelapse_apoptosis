#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# In[1]:


import argparse

# set the gpu via OS environment variable
import os
import pathlib

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
import torch
from cellpose import io as cellpose_io
from cellpose import models
from csbdeep.utils import normalize
from PIL import Image
from stardist.plot import render_label

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(in_notebook)
# check if we have a GPU
use_gpu = torch.cuda.is_available()
print("GPU activated:", use_gpu)


# In[2]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    parser.add_argument(
        "--clip_limit",
        type=float,
        help="Clip limit for the adaptive histogram equalization",
    )
    parser.add_argument(
        "--diameter",
        type=bool,
        help="diameter of the nuclei to segment",
    )

    args = parser.parse_args()
    clip_limit = args.clip_limit
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    diameter = args.diameter

else:
    input_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/test_data/timelapse/20231017ChromaLive_6hr_4ch_MaxIP_C-02_F0001"
    ).resolve(strict=True)
    # input_dir = pathlib.Path(
    #     "../../2.cellprofiler_ic_processing/illum_directory/test_data/endpoint/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_E-11_F0001"
    # ).resolve(strict=True)
    clip_limit = 0.6
    diameter = 100


# In[3]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_before, total_before = torch.cuda.mem_get_info(device)
starting_level_GPU_RAM = (total_before - free_before) / 1024**2
print("Starting level of GPU RAM available (MB): ", starting_level_GPU_RAM)


# ## Set up images, paths and functions

# In[4]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[5]:


image_dict = {
    "nuclei_file_paths": [],
    "nuclei": [],
    "cytoplasm1": [],
    "cytoplasm2": [],
    "cytoplasm3": [],
}


# In[6]:


# split files by channel
if not "Annexin" in input_dir.name:
    for file in files:
        if "C01" in file.split("/")[-1]:
            image_dict["nuclei_file_paths"].append(file)
            image_dict["nuclei"].append(tifffile.imread(file).astype(np.float32))
        elif "C02" in file.split("/")[-1]:
            image_dict["cytoplasm1"].append(tifffile.imread(file).astype(np.float32))
        elif "C03" in file.split("/")[-1]:
            image_dict["cytoplasm2"].append(tifffile.imread(file).astype(np.float32))
        elif "C04" in file.split("/")[-1]:
            image_dict["cytoplasm3"].append(tifffile.imread(file).astype(np.float32))

    cytoplasm_image_list = [
        np.max(
            np.array(
                [
                    cytoplasm1,
                    cytoplasm2,
                    cytoplasm3,
                ]
            ),
            axis=0,
        )
        for cytoplasm1, cytoplasm2, cytoplasm3 in zip(
            image_dict["cytoplasm1"],
            image_dict["cytoplasm2"],
            image_dict["cytoplasm3"],
        )
    ]
else:
    for file in files:
        if "C01" in file.split("/")[-1]:
            image_dict["nuclei_file_paths"].append(file)
            image_dict["nuclei"].append(tifffile.imread(file).astype(np.float32))
        elif "C05" in file.split("/")[-1]:
            image_dict["cytoplasm1"].append(tifffile.imread(file).astype(np.float32))

    cytoplasm_image_list = image_dict["cytoplasm1"]


nuclei_image_list = [np.array(nuclei) for nuclei in image_dict["nuclei"]]

cyto = np.array(cytoplasm_image_list).astype(np.int16)
nuclei = np.array(nuclei_image_list).astype(np.int16)

cyto = skimage.exposure.equalize_adapthist(cyto, clip_limit=clip_limit + 0.3)
nuclei = skimage.exposure.equalize_adapthist(nuclei, clip_limit=clip_limit)


print(cyto.shape, nuclei.shape)


# In[7]:


original_nuclei_image = nuclei.copy()
original_cyto_image = cyto.copy()


# In[8]:


imgs = []
# save each z-slice as an RGB png
for z in range(cyto.shape[0]):

    nuclei_tmp = nuclei[z, :, :]
    cyto_tmp = cyto[z, :, :]
    nuclei_tmp = (nuclei_tmp / nuclei_tmp.max() * 255).astype(np.uint8)
    cyto_tmp = (cyto_tmp / cyto_tmp.max() * 255).astype(np.uint8)
    # save the image as an RGB png with nuclei in blue and cytoplasm in red
    RGB = np.stack([cyto_tmp, np.zeros_like(cyto_tmp), nuclei_tmp], axis=-1)

    # change to 8-bit
    RGB = (RGB / RGB.max() * 255).astype(np.uint8)

    rgb_image_pil = Image.fromarray(RGB)

    imgs.append(rgb_image_pil)
imgs = np.array(imgs)


# ## Cellpose

# Weird errors occur when running this converted notebook in the command line.
# This cell helps the python interpreter figure out where it is...somehow.

# In[9]:


test = imgs[0, :, :, :]
print(test.shape)
model_name = "cyto3"

model = models.Cellpose(model_type=model_name, gpu=True)

channels = [[1, 3]]

# # get masks
# for _ in range(1):
masks, flows, styles, diams = model.eval(test, channels=channels, diameter=diameter)


# In[10]:


# model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
model_name = "cyto3"
model = models.Cellpose(model_type=model_name, gpu=True)

channels = [[1, 3]]

masks_all_dict = {"masks": [], "imgs": []}

# get masks for all the images
# save to a dict for later use
for img in imgs:
    img = normalize(img)
    masks, flows, styles, diams = model.eval(img, channels=channels, diameter=diameter)

    masks_all_dict["masks"].append(masks)
    masks_all_dict["imgs"].append(img)
print(len(masks_all_dict))
masks_all = masks_all_dict["masks"]
imgs = masks_all_dict["imgs"]

masks_all = np.array(masks_all)
imgs = np.array(imgs)

for frame_index, frame in enumerate(image_dict["nuclei_file_paths"]):
    tifffile.imwrite(
        f"{input_dir}/{str(frame).split('/')[-1].split('_C01')[0]}_cell_mask.tiff",
        masks_all[frame_index, :, :],
    )


# In[11]:


if in_notebook:
    for z in range(len(masks_all)):
        plt.figure(figsize=(20, 10))
        plt.title(f"z: {z}")
        plt.axis("off")
        plt.subplot(1, 2, 1)
        plt.imshow(imgs[z], cmap="gray")
        plt.title("RGB: Red - Cytoplasm, Blue - Nuclei")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(render_label(masks_all[z]))
        plt.title("Cell masks")
        plt.axis("off")
        plt.show()


# In[12]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_after, total_after = torch.cuda.mem_get_info(device)
amount_used = ((total_after - free_after)) / 1024**2
print(f"Used: {amount_used} MB or {amount_used / 1024} GB of GPU RAM")
