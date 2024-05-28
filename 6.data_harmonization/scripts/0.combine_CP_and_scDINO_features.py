#!/usr/bin/env python
# coding: utf-8

# This notebook combines the CellProfiller extracted morphology features and the scDINO extracted morphology features into one feature space. Downstream notebooks will normalize the data and perform feature selection.

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# In[2]:


# define data paths for import
cellprofiller_annotated_features_path = pathlib.Path(
    "../../4.process_features/data/annotated_data/run_20231017ChromaLive_6hr_4ch_MaxIP_sc.parquet"
).resolve(strict=True)

scdino_features = pathlib.Path(
    "../../5.scDINO_analysis/1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve(strict=True)

# set the output path
output_path = pathlib.Path(
    "../data/20231017ChromaLive_6hr_4ch_MaxIP_combined_data.parquet"
).resolve()

# make the parent directory
output_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


# load in the data
cellprofiller_data = pd.read_parquet(cellprofiller_annotated_features_path)
scdino_data = pd.read_parquet(scdino_features)
print(f"CellProfiller data shape: {cellprofiller_data.shape}")
print(f"scDINO data shape: {scdino_data.shape}")
cellprofiller_data.head()


# In[4]:


scdino_data.head()


# In[5]:


# get metadata columns
metadata_cols = [
    col for col in cellprofiller_data.columns if col.startswith("Metadata_")
]
metadata_cp_df = cellprofiller_data[metadata_cols].copy()
metadata_cp_df = metadata_cp_df[metadata_cols]
metadata_cp_df.head(1)


# In[6]:


# get metadata columns
metadata_cols = [col for col in scdino_data.columns if col.startswith("Metadata_")]
metadata_scdino_df = scdino_data[metadata_cols].copy()
metadata_scdino_df = metadata_scdino_df[metadata_cols]
metadata_scdino_df.head(1)


# In[7]:


merge_columns = [
    "Metadata_Well",
    "Metadata_FOV",
    "Metadata_Time",
    "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number",
]

# make all of the merge columns in both dfs object types
for col in merge_columns:
    cellprofiller_data[col] = cellprofiller_data[col].astype(str)
    scdino_data[col] = scdino_data[col].astype(str)

# make all of the merge columns in both metadata dfs object types
for col in merge_columns:
    metadata_cp_df[col] = metadata_cp_df[col].astype(str)
    metadata_scdino_df[col] = metadata_scdino_df[col].astype(str)


# In[8]:


# make the merge columns the index
cellprofiller_data.set_index(merge_columns, inplace=True)
scdino_data.set_index(merge_columns, inplace=True)


# In[9]:


# get the unique index values
scdino_index = scdino_data.index
cellprofiller_index = cellprofiller_data.index

unique_scdino_index = scdino_index.nunique()
unique_cellprofiller_index = cellprofiller_index.nunique()
scdino_index_not_in_cp = scdino_index.difference(cellprofiller_index)
cp_index_not_in_scdino = cellprofiller_index.difference(scdino_index)
index_in_both = scdino_index.intersection(cellprofiller_index)

print(f"Unique scDINO index values: {scdino_index.nunique()}")
print(f"Unique CellProfiller index values: {cellprofiller_index.nunique()}")
print(
    f"scDINO index values not in CellProfiller: {scdino_index.difference(cellprofiller_index).nunique()}"
)
print(
    f"CellProfiller index values not in scDINO: {cellprofiller_index.difference(scdino_index).nunique()}"
)
print(
    f"Index values in both: {scdino_index.intersection(cellprofiller_index).nunique()}"
)

# print(f"Uique scDINO - CP index values: {unique_scdino_index - unique_cellprofiller_index}")
print(
    f"Unique CP - scDINO index values: {unique_cellprofiller_index - unique_scdino_index}"
)

scdino_data = scdino_data.loc[index_in_both]
cellprofiller_data = cellprofiller_data.loc[index_in_both]


# In[10]:


# concatenate the data
combined_data = pd.concat([cellprofiller_data, scdino_data], axis=1)
print(combined_data.shape)
# drop duplicate columns
combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
# reset the index
combined_data.reset_index(inplace=True)
combined_data.head()


# In[11]:


# save the combined data
combined_data.to_parquet(output_path)
