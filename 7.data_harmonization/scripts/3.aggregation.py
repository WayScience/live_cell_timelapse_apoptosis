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


# ## Perform aggregation

# In[3]:


metadata_cols = fs_df.columns[fs_df.columns.str.contains("Metadata")]
feature_cols = fs_df.columns[~fs_df.columns.str.contains("Metadata")].to_list()

aggregated_df = aggregate(
    fs_df,
    features=feature_cols,
    strata=["Metadata_Well", "Metadata_Time", "Metadata_dose"],
    operation="median",
)

print(aggregated_df.shape)
aggregated_df.to_parquet(output_profile_dir)
aggregated_df.head()
