#!/usr/bin/env python
# coding: utf-8

# # Aggregate feature selected profiles

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
from pycytominer import aggregate

# ## Set paths and variables

# In[2]:


# set paths
input_profile_dir = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
).resolve(strict=True)
output_profile_dir = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm_fs_aggregated.parquet"
).resolve()
fs_df = pd.read_parquet(input_profile_dir)
fs_df.head()


# ## Perform aggregation

# In[3]:


metadata_cols = fs_df.columns[fs_df.columns.str.contains("Metadata")].to_list()
feature_cols = fs_df.columns[~fs_df.columns.str.contains("Metadata")].to_list()
selected_metadata_cols = [
    "Metadata_Well",
    "Metadata_plate",
    "Metadata_compound",
    "Metadata_dose",
    "Metadata_control",
    "Metadata_Time",
]
feature_cols = fs_df.columns[~fs_df.columns.str.contains("Metadata")].to_list()
feature_cols = ["Metadata_number_of_singlecells"] + feature_cols

aggregated_df = aggregate(
    fs_df,
    features=feature_cols,
    strata=["Metadata_Well", "Metadata_Time", "Metadata_dose"],
    operation="median",
)
aggregated_df = pd.merge(
    aggregated_df,
    fs_df[selected_metadata_cols],
    how="left",
    on=["Metadata_Well", "Metadata_Time", "Metadata_dose"],
)
aggregated_df.drop_duplicates(inplace=True, ignore_index=True)

# rearrange the columns such that the metadata columns are first
for col in reversed(aggregated_df.columns):
    if col.startswith("Metadata_"):
        tmp_pop = aggregated_df.pop(col)
        aggregated_df.insert(0, col, tmp_pop)

print(aggregated_df.shape)
aggregated_df.to_parquet(output_profile_dir)
aggregated_df.head()
