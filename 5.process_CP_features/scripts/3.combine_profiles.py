#!/usr/bin/env python
# coding: utf-8

# # Normalize annotated single cells using negative control

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd

# ## Set paths and variables

# In[2]:


# set paths
paths_dict = {
    "timelapse_profiles": {
        "input_dir": pathlib.Path("../data/2.sc_tracks_annotated_data/profiles"),
        "output_file_dir": pathlib.Path(
            "../data/3.combined_data/profiles/combined_data.parquet"
        ),
    },
    "endpoint_data": {
        "input_dir": pathlib.Path("../data/1.annotated_data/endpoint"),
        "output_file_dir": pathlib.Path(
            "../data/3.combined_data/endpoints/combined_data.parquet"
        ),
    },
}


# In[3]:


for data_set in paths_dict.keys():
    files = list(paths_dict[data_set]["input_dir"].rglob("*.parquet"))
    print(f"Found {len(files)} files in {data_set} directory")
    list_of_dfs = [pd.read_parquet(file) for file in files]
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    paths_dict[data_set]["output_file_dir"].parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(paths_dict[data_set]["output_file_dir"])
    print(combined_df.shape)
