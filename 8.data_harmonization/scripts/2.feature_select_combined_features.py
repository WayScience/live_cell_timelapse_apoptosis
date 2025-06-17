#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd
from pycytominer import feature_select
from pycytominer.cyto_utils import output

# In[2]:


# set path to normalized data
normalized_data_path = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm.parquet"
).resolve(strict=True)

# set the outout file path
feature_selected_output_file_path = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
).resolve()

# read in the normalized data
normalized_data = pd.read_parquet(normalized_data_path)
print(normalized_data.shape)
normalized_data.head()


# In[3]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]

# Get columns that contain "Metadata"
metadata_features = normalized_data.columns[
    normalized_data.columns.str.contains("Metadata")
].tolist()

# get the feature columns
feature_columns = normalized_data.columns.difference(metadata_features).to_list()


# In[4]:


manual_block_list = [
    "Nuclei_AreaShape_BoundingBoxArea",
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Cells_AreaShape_BoundingBoxArea",
]


# In[ ]:


feature_select_df = feature_select(
    normalized_data,
    operation=feature_select_ops,
    features=feature_columns,
)
# add "Metadata_" to the beginning of each column name in the list
feature_select_df.columns = [
    "Metadata_" + column if column in manual_block_list else column
    for column in feature_select_df.columns
]
print("Feature selection complete, saving to parquet file!")
# save features selected df as parquet file
output(
    df=feature_select_df,
    output_filename=feature_selected_output_file_path,
    output_type="parquet",
)
print("Features have been selected!")
# check to see if the shape of the df has changed indicating feature selection occurred
print(normalized_data.shape)
print(feature_select_df.shape)
print(f"{normalized_data.shape[1] - feature_select_df.shape[1]} features were removed.")
print(f"{feature_select_df.shape[1]} features remain.")
feature_select_df.head()
