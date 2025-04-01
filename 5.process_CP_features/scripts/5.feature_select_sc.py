#!/usr/bin/env python
# coding: utf-8

# # Perform feature selection on normalized data

# ## Import libraries

# In[1]:


import gc
import pathlib

import pandas as pd
from pycytominer import feature_select
from pycytominer.cyto_utils import output

# ## Set paths and variables

# In[2]:


# set paths
paths_dict = {
    "timelapse_profiles": {
        "input_dir": pathlib.Path(
            "../data/4.normalized_data/profiles/normalized_profile.parquet"
        ).resolve(strict=True),
        "output_file_dir": pathlib.Path(
            "../data/5.feature_select/profiles/features_selected_profile.parquet"
        ).resolve(),
    },
    "endpoint_data": {
        "input_dir": pathlib.Path(
            "../data/4.normalized_data/endpoints/normalized_profile.parquet"
        ).resolve(strict=True),
        "output_file_dir": pathlib.Path(
            "../data/5.feature_select/endpoints/features_selected_profile.parquet"
        ).resolve(),
    },
}


# ## Perform feature selection

# In[3]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]


# In[4]:


manual_block_list = [
    "Nuclei_AreaShape_BoundingBoxArea",
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Cells_AreaShape_BoundingBoxArea",
]


# In[5]:


for data_set in paths_dict:
    paths_dict[data_set]["output_file_dir"].parent.mkdir(exist_ok=True, parents=True)
    # read in the annotated file
    normalized_df = pd.read_parquet(paths_dict[data_set]["input_dir"])
    # perform feature selection with the operations specified
    feature_select_df = feature_select(
        normalized_df,
        operation=feature_select_ops,
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
        output_filename=paths_dict[data_set]["output_file_dir"],
        output_type="parquet",
    )
    # check to see if the shape of the df has changed indicating feature selection occurred
    print(feature_select_df.shape)
    feature_select_df.head()
