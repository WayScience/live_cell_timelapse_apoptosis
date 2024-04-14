#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# In[2]:


# set paths
# input data
cls_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/test_run/CLS_features/channel_DNA_channel488-1_channel488-2_channel561_blank_model_sc-ViT_checkpoint0100_vitsmall16_features.csv"
).resolve(strict=True)
image_paths_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/test_run/CLS_features/image_paths.csv"
).resolve(strict=True)

# output path for the merged table
output_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/test_run/CLS_features/CLS_features_annotated.parquet"
).resolve()


# In[3]:


# load data
cls_df = pd.read_csv(cls_path, header=None)
image_paths_df = pd.read_csv(image_paths_path, header=None)


# In[4]:


print(len(cls_df), len(image_paths_df))


# In[5]:


# rename each column to have cls_ prefix
cls_df.columns = [f"cls_{i}" for i in range(cls_df.shape[1])]
# rename the columns
image_paths_df.columns = ["image_paths"]


# In[6]:


# combine data
cls_df["Metadata_image_path"] = image_paths_df["image_paths"]
cls_df.head(1)


# In[7]:


# split column into multiple columns
cls_df["Metadata_Well"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[0]
)
cls_df["Metadata_FOV"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[1]
)
cls_df["Metadata_Time"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[2]
)
cls_df["Metadata_Channel"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[4]
)
cls_df["Metadata_Cell_id"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[6]
)
# move Metadata columns to the front
# pop out the Metadata columns
Metadata_cols = cls_df.columns[cls_df.columns.str.contains("Metadata")]
# move Metadata columns to the front
cls_df = cls_df[
    Metadata_cols.tolist() + cls_df.columns.difference(Metadata_cols).tolist()
]


cls_df.head()


# In[8]:


# save the data
cls_df.to_parquet(output_path, index=False)
