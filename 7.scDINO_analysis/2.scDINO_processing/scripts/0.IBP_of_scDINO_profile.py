#!/usr/bin/env python
# coding: utf-8

# This notebook performs IBP on the scDINO profile data.
# We perform the following steps:
# 1. Load the scDINO profile data.
# 2. Normalize the data.
# 3. Feature select the data.
# 4. Aggregate the data.

# In[1]:


import pathlib

import pandas as pd
from pycytominer import aggregate, feature_select, normalize

# In[2]:


scDINO_profile_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve(strict=True)

scDINO_normalized_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated_normalized.parquet"
).resolve()
scDINO_feature_selected_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated_normalized_feature_selected.parquet"
).resolve()
scDINO_aggregated_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated_normalized_feature_selected_aggregated.parquet"
).resolve()

scDINO_profile = pd.read_parquet(scDINO_profile_path)
print(f"scDINO profile shape: {scDINO_profile.shape}")
scDINO_profile.head()


# ## Normalization

# In[3]:


metadata_columns = [x for x in scDINO_profile.columns if "metadata" in x.lower()]
features = [x for x in scDINO_profile.columns if x not in metadata_columns]
normalized_df = normalize(
    # df with annotated raw merged single cell features
    profiles=scDINO_profile,
    # specify samples used as normalization reference (negative control)
    samples="Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0",
    # normalization method used
    method="standardize",
    features=features,
    meta_features=metadata_columns,
)

normalized_df.to_parquet(
    scDINO_normalized_path,
    index=False,
)
# check to see if the features have been normalized
print(normalized_df.shape)
normalized_df.head()


# ## Feature Selection

# In[4]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]
manual_block_list = [
    x
    for x in normalized_df.columns
    if "bounding" in x.lower()
    or "Location_Center_Y" in x.lower()
    or "Location_Center_X" in x.lower()
]
metadata_columns = [x for x in normalized_df.columns if "metadata" in x.lower()]
features = [
    x for x in normalized_df.columns if x not in metadata_columns + manual_block_list
]

# perform feature selection with the operations specified
feature_select_df = feature_select(
    normalized_df,
    operation=feature_select_ops,
    features=features,
)
# merge the metadata columns back into the feature selected dataframe
feature_select_df = pd.merge(
    feature_select_df,
    normalized_df[metadata_columns],
    how="left",
)
print(f"Number of features before feature selection: {len(features)}")
print(f"Number of features after feature selection: {len(feature_select_df.columns)}")
feature_select_df.to_parquet(
    scDINO_feature_selected_path,
    index=False,
)
feature_select_df.head()


# ## Aggregation

# In[5]:


metadata_cols = [
    "Metadata_Well",
    "Metadata_Time",
    "Metadata_compound",
    "Metadata_dose",
    "Metadata_control",
]
feature_cols = feature_select_df.columns[
    ~feature_select_df.columns.str.contains("Metadata")
].to_list()
aggregated_df = aggregate(
    feature_select_df,
    features=feature_cols,
    strata=["Metadata_Well", "Metadata_Time"],
    operation="median",
)
aggregated_df = pd.merge(
    aggregated_df,
    feature_select_df[metadata_cols],
    how="left",
    on=["Metadata_Well", "Metadata_Time"],
)
aggregated_df.drop_duplicates(
    subset=metadata_cols,
    inplace=True,
)
aggregated_df.reset_index(drop=True, inplace=True)
aggregated_df["Metadata_Time"] = (
    aggregated_df["Metadata_Time"].astype(int) - 1
)  # adjust for 0-based indexing
print(f"Number of samples before aggregation: {feature_select_df.shape[0]}")
print(f"Number of samples after aggregation: {aggregated_df.shape[0]}")
aggregated_df.to_parquet(
    scDINO_aggregated_path,
    index=False,
)
aggregated_df.head()
