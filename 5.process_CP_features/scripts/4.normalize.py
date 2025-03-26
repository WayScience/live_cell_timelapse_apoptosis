#!/usr/bin/env python
# coding: utf-8

# # Normalize annotated single cells using negative control

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
from pycytominer import normalize
from pycytominer.cyto_utils import output

# ## Set paths and variables

# In[2]:


# set paths
paths_dict = {
    "timelapse_profiles": {
        "input_dir": pathlib.Path(
            "../data/3.combined_data/profiles/combined_data.parquet"
        ),
        "outout_file_dir": pathlib.Path(
            "../data/4.normalized_data/profiles/normalized_profile.parquet"
        ),
    },
    "endpoint_data": {
        "input_dir": pathlib.Path(
            "../data/3.combined_data/endpoints/combined_data.parquet"
        ),
        "outout_file_dir": pathlib.Path(
            "../data/4.normalized_data/endpoints/normalized_profile.parquet"
        ),
    },
}


# ## Normalize with standardize method with negative control on annotated data

# The normalization needs to occur per time step.
# This code cell will split the data into time steps and normalize each time step separately.
# Then each normalized time step will be concatenated back together.

# In[3]:


for data_set in paths_dict:
    # read data
    paths_dict[data_set]["outout_file_dir"].parent.mkdir(exist_ok=True, parents=True)
    annotated_df = pd.read_parquet(paths_dict[data_set]["input_dir"])
    # read in the annotated file
    annotated_df.reset_index(drop=True, inplace=True)
    Metadatas = annotated_df.columns[
        annotated_df.columns.str.contains("Metadata")
    ].to_list()
    features = annotated_df.columns[~annotated_df.columns.isin(Metadatas)].to_list()
    # normalize annotated data
    if data_set not in "endpoint_data":
        normalized_df = normalize(
            # df with annotated raw merged single cell features
            profiles=annotated_df,
            # specify samples used as normalization reference (negative control)
            samples="Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0 and Metadata_Time == 0.0",
            # normalization method used
            method="standardize",
            features=features,
            meta_features=Metadatas,
        )
    else:
        normalized_df = normalize(
            # df with annotated raw merged single cell features
            profiles=annotated_df,
            # specify samples used as normalization reference (negative control)
            samples="Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0",
            # normalization method used
            method="standardize",
            features=features,
            meta_features=Metadatas,
        )

    output(
        normalized_df,
        output_filename=paths_dict[data_set]["outout_file_dir"],
        output_type="parquet",
    )
    # check to see if the features have been normalized
    print(normalized_df.shape)
    normalized_df.head()
