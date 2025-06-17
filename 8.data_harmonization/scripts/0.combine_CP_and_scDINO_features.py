#!/usr/bin/env python
# coding: utf-8

# This notebook combines the cellprofiler extracted morphology features and the scDINO extracted morphology features into one feature space. Downstream notebooks will normalize the data and perform feature selection.

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# In[ ]:


# define data paths for import
# annotated features from cellprofiler including all time points
cellprofiler_fs_features_path = pathlib.Path(
    "../../6.process_CP_features/data/5.feature_select/profiles/features_selected_profile.parquet"
).resolve(strict=True)

# scDINO features from the scDINO analysis including all time points
scdino_features = pathlib.Path(
    "../../7.scDINO_analysis/1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve(strict=True)

# set the output path
output_path = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_data.parquet"
).resolve()

# make the parent directory
output_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


# load in the data
cellprofiler_data = pd.read_parquet(cellprofiler_fs_features_path)
scdino_data = pd.read_parquet(scdino_features)

print(f"cellprofiler data shape: {cellprofiler_data.shape}")
print(f"scDINO data shape: {scdino_data.shape}")


# In[4]:


cellprofiler_data["Metadata_original_index"] = cellprofiler_data.index


# In[5]:


scdino_data.head(1)


# In[6]:


# append either CP or scDINO to the column names
for col in cellprofiler_data.columns:
    # ensure Metadata is not in the column name
    if not "Metadata" in col:
        cellprofiler_data.rename(columns={col: f"{col}_CP"}, inplace=True)
for col in scdino_data.columns:
    # ensure Metadata is not in the column name
    if not "Metadata" in col:
        scdino_data.rename(columns={col: f"{col}_scDINO"}, inplace=True)


# In[7]:


# make the Metadata Columns objects
# these are the columns that are common between the two datasets
cellprofiler_metadata_columns = [
    "Metadata_Well",
    "Metadata_FOV",
    "Metadata_Time",
    "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number",
    "Metadata_compound",
    "Metadata_dose",
    "Metadata_control",
    "Metadata_original_index",
]


# In[8]:


scdino_data.head()
# convert time to float
scdino_data["Metadata_Time"] = scdino_data["Metadata_Time"].astype(float)
scdino_data["Metadata_Time"] = scdino_data["Metadata_Time"] - 1
scdino_data.head()


# In[9]:


for col in cellprofiler_metadata_columns:
    if col not in cellprofiler_data.columns:
        raise ValueError(f"{col} not found in cellprofiler data.")
    cellprofiler_data[col] = cellprofiler_data[col].astype(str)
    if col not in scdino_data.columns:
        raise ValueError(f"{col} not found in scDINO data.")
    scdino_data[col] = scdino_data[col].astype(str)


# In[10]:


print(f"cellprofiler data shape after sorting: {cellprofiler_data.shape}")
print(f"scDINO data shape after sorting: {scdino_data.shape}")
merged_df = pd.merge(
    cellprofiler_data,
    scdino_data,
    how="inner",
    on=cellprofiler_metadata_columns,
)
print(f"merged data shape: {merged_df.shape}")
# drop duplicates
merged_df = merged_df.drop_duplicates(
    subset=cellprofiler_metadata_columns,
    keep="last",
)
print(f"merged data shape after dropping duplicates: {merged_df.shape}")


# In[11]:


# merged_df.to_parquet(output_path)
print(f"merged_df shape: {merged_df.shape}")
# merged_df.head()
# drop rows with NaN values
merged_df = merged_df.dropna(how="any", axis=0)
merged_df.to_parquet(output_path, index=False)
print(f"merged_df shape: {merged_df.shape}")
merged_df.head()
