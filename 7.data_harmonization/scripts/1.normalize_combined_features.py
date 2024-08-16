#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd
from pycytominer import normalize
from pycytominer.cyto_utils import output

# In[2]:


# set path to data

combined_data_path = pathlib.Path(
    "../data/20231017ChromaLive_6hr_4ch_MaxIP_combined_data.parquet"
).resolve(strict=True)

# set output path
normalized_data_output_path = pathlib.Path(
    "../data/20231017ChromaLive_6hr_4ch_MaxIP_normalized_combined_data.parquet"
).resolve()

# load data
combined_data = pd.read_parquet(combined_data_path)
print(combined_data.shape)
combined_data.head()


# In[3]:


# if column name contains TrackObjects, then prepend with Metadata
combined_data.columns = [
    "Metadata_" + x if "TrackObjects" in x else x for x in combined_data.columns
]


# In[4]:


# Get columns that contain "Metadata"
metadata_features = combined_data.columns[
    combined_data.columns.str.contains("Metadata")
].tolist()

# get the feature columns
feature_columns = combined_data.columns.difference(metadata_features).to_list()


# In[5]:


# Normalize the single cell data per time point

# make the time column an integer
combined_data.Metadata_Time = combined_data.Metadata_Time.astype(int)

# get the unique time points
time_points = combined_data.Metadata_Time.unique()

output_dict_of_normalized_dfs = {}

# define a for loop to normalize each time point
for time_point in time_points:
    # subset the data to the time point
    time_point_df = combined_data.loc[combined_data.Metadata_Time == time_point]

    # normalize annotated data
    normalized_df = normalize(
        # df with annotated raw merged single cell features
        profiles=time_point_df,
        features=feature_columns,
        meta_features=metadata_features,
        # specify samples used as normalization reference (negative control)
        samples=f"Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0 and Metadata_Time == {time_point}",
        # normalization method used
        method="standardize",
    )

    output_dict_of_normalized_dfs[time_point] = normalized_df

# combine the normalized dataframes
normalized_df = pd.concat(output_dict_of_normalized_dfs.values()).reset_index(drop=True)

output(
    normalized_df,
    output_filename=normalized_data_output_path,
    output_type="parquet",
)
print(f"Single cells have been normalized!")
# check to see if the features have been normalized
print(normalized_df.shape)
normalized_df.head()
