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
    "../../5.process_CP_features/data/3.combined_data/profiles/combined_data.parquet"
).resolve(strict=True)

scdino_features = pathlib.Path(
    "../../6.scDINO_analysis/1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve(strict=True)

# set the output path
output_path = pathlib.Path("../data/CP_scDINO_features/combined_data.parquet").resolve()

# make the parent directory
output_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


# load in the data
cellprofiller_data = pd.read_parquet(cellprofiller_annotated_features_path)
scdino_data = pd.read_parquet(scdino_features)

print(f"CellProfiller data shape: {cellprofiller_data.shape}")
print(f"scDINO data shape: {scdino_data.shape}")


# In[4]:


# append either CP or scDINO to the column names
for col in cellprofiller_data.columns:
    # ensure Metadata is not in the column name
    if not "Metadata" in col:
        cellprofiller_data.rename(columns={col: f"{col}_CP"}, inplace=True)
for col in scdino_data.columns:
    # ensure Metadata is not in the column name
    if not "Metadata" in col:
        scdino_data.rename(columns={col: f"{col}_scDINO"}, inplace=True)


# In[5]:


# make the Metadata Columns objects
cellprofiler_metadata_columns = [
    "Metadata_Well",
    "Metadata_FOV",
    "Metadata_Time",
    "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number",
    "Metadata_compound",
    "Metadata_dose",
    "Metadata_control",
]


# In[ ]:


# sort the data by Well, FOV, Time, ImageNumber, Nuclei_Number_Object_Number
cellprofiller_data = cellprofiller_data.sort_values(
    by=cellprofiler_metadata_columns,
    ascending=True,
).reset_index(drop=True)

scdino_data = scdino_data.sort_values(
    by=cellprofiler_metadata_columns,
    ascending=True,
).reset_index(drop=True)


# In[7]:


scdino_data.head()
# convert time to float
scdino_data["Metadata_Time"] = scdino_data["Metadata_Time"].astype(float)
scdino_data["Metadata_Time"] = scdino_data["Metadata_Time"] - 1
scdino_data.head()


# In[8]:


for col in cellprofiler_metadata_columns:
    if col not in cellprofiller_data.columns:
        raise ValueError(f"{col} not found in CellProfiler data.")
    cellprofiller_data[col] = cellprofiller_data[col].astype(str)
    if col not in scdino_data.columns:
        raise ValueError(f"{col} not found in scDINO data.")
    scdino_data[col] = scdino_data[col].astype(str)


# In[9]:


merged_df = pd.merge(
    cellprofiller_data,
    scdino_data,
    how="inner",
    # on=cellprofiler_metadata_columns,
    on=cellprofiler_metadata_columns,
)


# In[10]:


merged_df.to_parquet(output_path)
print(f"merged_df shape: {merged_df.shape}")
merged_df.head()
