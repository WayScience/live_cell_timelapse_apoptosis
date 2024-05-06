#!/usr/bin/env python
# coding: utf-8

# This notebook converts single channel grayscale images to 5 channel images by adding blank channels.
# This is done to make the images compatible with the pre-trained models that expect 5 channel images.
# The code in this notebook will need to change to match a unique dataset, regretfully.
#
# Note that the data used here has four channels, but the model needs 5 channels input.

# In[ ]:


import pathlib

import cv2

# show the image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

# ## Import paths

# In[ ]:


# set the path to the data directory
data_file_dir = pathlib.Path(
    "../../data/extracted_features/run_20231017ChromaLive_6hr_4ch_MaxIP_sc.parquet"
).resolve(strict=True)

# read in the data
cp_feature_data = pd.read_parquet(data_file_dir)
# print the data
print(cp_feature_data.shape)
cp_feature_data.head()


# In[ ]:


# rename Columns that contain Image to start with Metadata_
cp_feature_data = cp_feature_data.rename(
    columns=lambda x: x if not "Name" in x else "Metadata_" + x
)
# rename Columns that contain BoundingBox to start with Metadata_
cp_feature_data = cp_feature_data.rename(
    columns=lambda x: x if not "BoundingBox" in x else "Metadata_" + x
)
# rename Columns that contain Center_ to start with Metadata_
cp_feature_data = cp_feature_data.rename(
    columns=lambda x: x if not "Center_" in x else "Metadata_" + x
)
# get columns that contain Metadata
metadata_columns = [col for col in cp_feature_data.columns if "Metadata" in col]
metadata_df = cp_feature_data[metadata_columns]
# get columns that contain Features
feature_df = cp_feature_data.drop(columns=metadata_columns)
metadata_df.head()


# Metadata_Image_FileName_488_1
# Metadata_Image_FileName_488_2
# Metadata_Image_FileName_561
# Metadata_Image_FileName_DNA

# In[ ]:


# define the center x and y and path
total_counter = 0
ommited_counter = 0
# define the psuedo radius
radius = 50
for i in range(len(metadata_df)):
    total_counter += 1
    image_information_df = metadata_df.iloc[i]
    image_path = pathlib.Path(
        "../../../live_cell_timelapse_apoptosis/2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_6hr_4ch_MaxIP/"
    )
    center_y = image_information_df["Metadata_Nuclei_Location_Center_Y"].astype(int)
    center_x = image_information_df["Metadata_Nuclei_Location_Center_X"].astype(int)
    # DNA
    image_name_DNA = pathlib.Path(image_information_df["Metadata_Image_FileName_DNA"])
    image_path_DNA = pathlib.Path(image_path / image_name_DNA).resolve(strict=True)
    # 488_1
    image_name_488_1 = pathlib.Path(
        image_information_df["Metadata_Image_FileName_488_1"]
    )
    image_path_488_1 = pathlib.Path(image_path / image_name_488_1).resolve(strict=True)
    # 488_2
    image_name_488_2 = pathlib.Path(
        image_information_df["Metadata_Image_FileName_488_2"]
    )
    image_path_488_2 = pathlib.Path(image_path / image_name_488_2).resolve(strict=True)
    # 561
    image_name_561 = pathlib.Path(image_information_df["Metadata_Image_FileName_561"])
    image_path_561 = pathlib.Path(image_path / image_name_561).resolve(strict=True)
    image_DNA = tiff.imread(image_path_DNA)
    image_488_1 = tiff.imread(image_path_488_1)
    image_488_2 = tiff.imread(image_path_488_2)
    image_561 = tiff.imread(image_path_561)

    image_DNA_crop = image_DNA[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ]
    image_488_1_crop = image_488_1[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ]
    image_488_2_crop = image_488_2[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ]
    image_561_crop = image_561[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ]

    # check if crop is an edge case
    # Where edge case is cells that are too close to the edge of the image to crop
    # This ensures that all crops are the same dimensions and can be used in the model
    if image_DNA_crop.shape[0] < radius * 2 or image_DNA_crop.shape[1] < radius * 2:
        print(
            f"Image {image_information_df['Metadata_Image_FileName_DNA']} is an edge case. Ommitting..."
        )
        ommited_counter += 1
        continue
    # merge the channels to a single image
    image_merge = np.stack(
        [image_DNA_crop, image_488_1_crop, image_488_2_crop, image_561_crop], axis=-1
    )
    if image_merge.shape[-1] < 5:
        channels_to_add = 5 - image_merge.shape[-1]
        for channel in range(channels_to_add):
            # add a new channel of all zeros
            new_channels = np.zeros((image_merge.shape[0], image_merge.shape[1], 1))
            image_merge = np.concatenate((image_merge, new_channels), axis=-1)
    print(image_merge.shape)
    # save images to disk
    image_save_path = pathlib.Path(
        f"../../data/processed_images/crops/{image_information_df['Metadata_Well']}"
    )
    image_save_path.mkdir(parents=True, exist_ok=True)
    file_name = image_information_df["Metadata_Image_FileName_DNA"].replace(
        ".tiff",
        f'_{image_information_df["Metadata_Nuclei_Number_Object_Number"]}_crop.tiff',
    )
    image_save_path = pathlib.Path(image_save_path / file_name)
    tiff.imwrite(image_save_path, image_merge)
    print(f"Image saved to {image_save_path}")


# In[ ]:


print(f"Total cell images: {total_counter}")
print(f"Ommited cell images: {ommited_counter}")
print(f"Total saved cell images: {total_counter - ommited_counter}")
print(
    f"{round(((total_counter - ommited_counter)/total_counter*100),2)}% of the images were saved"
)
