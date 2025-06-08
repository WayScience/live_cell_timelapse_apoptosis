#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# In[2]:


# set paths
# input data
cls_path_dict = {
    "channel_DNA": pathlib.Path(
        "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/channel_DNA_model_dino_deitsmall16_pretrain_full_checkpoint_features.csv"
    ).resolve(strict=True),
    "channel488-1": pathlib.Path(
        "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/channel_channel488-1_model_dino_deitsmall16_pretrain_full_checkpoint_features.csv"
    ).resolve(strict=True),
    "channel488-2": pathlib.Path(
        "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/channel_channel488-2_model_dino_deitsmall16_pretrain_full_checkpoint_features.csv"
    ).resolve(strict=True),
    "channel561": pathlib.Path(
        "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/channel_channel561_model_dino_deitsmall16_pretrain_full_checkpoint_features.csv"
    ).resolve(strict=True),
}

# image paths
image_paths_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/image_paths.csv"
).resolve(strict=True)

# plate map
plate_map_path = pathlib.Path("../../../data/platemap_6hr_4ch.csv").resolve(strict=True)

# output path for the merged table
output_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve()


# In[3]:


cls_df_dict = {}
cls_df_dict["channel488-1_df"] = pd.read_csv(cls_path_dict["channel488-1"], header=None)
cls_df_dict["channel488-2_df"] = pd.read_csv(cls_path_dict["channel488-2"], header=None)
cls_df_dict["channel561_df"] = pd.read_csv(cls_path_dict["channel561"], header=None)
cls_df_dict["channel_DNA_df"] = pd.read_csv(cls_path_dict["channel_DNA"], header=None)


# In[4]:


image_paths_df = pd.read_csv(image_paths_path, header=None)
plate_map_df = pd.read_csv(plate_map_path)


# In[5]:


print(
    len(cls_df_dict["channel488-1_df"]),
    len(cls_df_dict["channel488-2_df"]),
    len(cls_df_dict["channel561_df"]),
    len(cls_df_dict["channel_DNA_df"]),
    len(image_paths_df),
)
print(
    cls_df_dict["channel488-1_df"].shape,
    cls_df_dict["channel488-2_df"].shape,
    cls_df_dict["channel561_df"].shape,
    cls_df_dict["channel_DNA_df"].shape,
    image_paths_df.shape,
)
print(plate_map_df.shape)


# In[6]:


# loop through each channel and adjust the cls_df column names
for channel, cls_df in cls_df_dict.items():
    channel = channel.strip("_df")
    cls_df.columns = [f"{channel}_cls_feature_{i}" for i in range(cls_df.shape[1])]

# merge the cls dataframes
cls_df = pd.concat(cls_df_dict.values(), axis=1)
cls_df.head()


# In[7]:


# rename the columns
image_paths_df.columns = ["image_paths"]


# In[8]:


# combine data
cls_df["Metadata_image_path"] = image_paths_df["image_paths"]
cls_df.head(1)


# In[15]:


pathlib.Path(cls_df["Metadata_image_path"][0]).name.split("cell_number_")[1].split(
    "_crop"
)[0].split("_index")[0]


# In[ ]:


# In[19]:


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

cls_df["Metadata_ImageNumber"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x).name.split("_")[7]
)

cls_df["Metadata_Nuclei_Number_Object_Number"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x)
    .name.split("cell_number_")[1]
    .split("_crop")[0]
    .split("_index")[0]
)

cls_df["Metadata_original_index"] = cls_df["Metadata_image_path"].apply(
    lambda x: pathlib.Path(x)
    .name.split("cell_number_")[1]
    .split("_crop")[0]
    .split("index_")[1]
)
cls_df.head()


# In[20]:


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


# In[21]:


# merge cls_df with plate_map_df
cls_df = cls_df.merge(plate_map_df, how="left", on="Metadata_Well")
Metadata_cols = cls_df.columns[cls_df.columns.str.contains("Metadata")]
# move Metadata columns to the front
cls_df = cls_df[
    Metadata_cols.tolist() + cls_df.columns.difference(Metadata_cols).tolist()
]
print(cls_df.shape)
cls_df.head()


# In[22]:


# remove the "F" from each value in the Metadata_FOV column
cls_df["Metadata_FOV"] = cls_df["Metadata_FOV"].str.replace("F", "")

# remove the "T" from each value in the Metadata_Time column
cls_df["Metadata_Time"] = cls_df["Metadata_Time"].str.replace("T", "")


# In[23]:


# save the data
cls_df.to_parquet(output_path)


# In[24]:


cls_df.head()


# In[ ]:


pd.set_option("display.max_columns", None)
cls_df
