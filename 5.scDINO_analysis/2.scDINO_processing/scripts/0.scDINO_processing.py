#!/usr/bin/env python
# coding: utf-8

# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [2]</a>'.</span>

# In[5]:


import pathlib

import numpy as np
import pandas as pd

# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# In[6]:


# set paths
# input data
cls_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/channel_DNA_channel488-1_channel488-2_channel561_blank_model_sc-ViT_checkpoint0100_vitsmall16_features.csv"
).resolve(strict=True)
image_paths_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/image_paths.csv"
).resolve(strict=True)

# plate map
plate_map_path = pathlib.Path("../../../data/platemap_6hr_4ch.csv").resolve(strict=True)

# output path for the merged table
output_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/testapoptosis_timelapse_run/CLS_features/CLS_features_annotated.parquet"
).resolve()


# In[ ]:


# load data
cls_df = pd.read_csv(cls_path, header=None)
image_paths_df = pd.read_csv(image_paths_path, header=None)
plate_map_df = pd.read_csv(plate_map_path)


# In[ ]:


print(len(cls_df), len(image_paths_df))


# In[ ]:


# rename each column to have cls_ prefix
cls_df.columns = [f"cls_{i}" for i in range(cls_df.shape[1])]
# rename the columns
image_paths_df.columns = ["image_paths"]


# In[ ]:


# combine data
cls_df["Metadata_image_path"] = image_paths_df["image_paths"]
cls_df.head(1)


# In[ ]:


# split column into multiple columns
# Well, FOV, Time, Channel, Cell_id
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

cls_df.head()


# In[ ]:


# drop the plate column
plate_map_df.drop(columns=["plate"], inplace=True)
# rename columns
plate_map_df = plate_map_df.rename(
    columns={
        "well": "Metadata_Well",
        "compound": "Metadata_compound",
        "dose": "Metadata_dose",
        "control": "Metadata_control",
    },
)
plate_map_df.head()


# In[ ]:


# merge cls_df with plate_map_df
cls_df = cls_df.merge(plate_map_df, how="left", on="Metadata_Well")
Metadata_cols = cls_df.columns[cls_df.columns.str.contains("Metadata")]
# move Metadata columns to the front
cls_df = cls_df[
    Metadata_cols.tolist() + cls_df.columns.difference(Metadata_cols).tolist()
]
cls_df.head()


# In[ ]:


# save the data
cls_df.to_parquet(output_path, index=False)
