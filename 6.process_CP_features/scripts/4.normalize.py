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

    # keep only rows where (well, fov, track_id, time) occurs more than twice,
    group_cols = [
        "Metadata_Well",
        "Metadata_FOV",
        "Metadata_track_id",
    ]
    time_cols = group_cols + ["Metadata_Time"]

    long_tracks_df = normalized_df[
        normalized_df.groupby(group_cols)["Metadata_track_id"].transform("size") > 13
    ]

    dup_idx = long_tracks_df.groupby(time_cols).size().loc[lambda s: s > 1].index
    long_tracks_df[long_tracks_df.set_index(time_cols).index.isin(dup_idx)]
    # pick one of the duplicated rows to keep and drop the row from the original df
    to_drop_idx = long_tracks_df[
        long_tracks_df.set_index(time_cols).index.isin(dup_idx)
    ].index
    original_df_shape = normalized_df.shape
    normalized_df = normalized_df.drop(index=to_drop_idx)
    print(
        f"Dropped {original_df_shape[0] - normalized_df.shape[0]} duplicated objects rows from the original df"
    )

    output(
        normalized_df,
        output_filename=paths_dict[data_set]["outout_file_dir"],
        output_type="parquet",
    )
    # check to see if the features have been normalized
    print(normalized_df.shape)
    normalized_df.head()


# In[12]:


group_cols = [
    "Metadata_Well",
    "Metadata_FOV",
    "Metadata_track_id",
]
time_cols = group_cols + ["Metadata_Time"]

long_tracks_df = normalized_df[
    normalized_df.groupby(group_cols)["Metadata_track_id"].transform("size") > 1
]
long_tracks_df["Metadata_track_id"]


# In[ ]:


dup_idx = long_tracks_df.groupby(time_cols).size().loc[lambda s: s > 1].index
long_tracks_df[long_tracks_df.set_index(time_cols).index.isin(dup_idx)]
# pick one of the duplicated rows to keep and drop the row from the original df
to_drop_idx = long_tracks_df[
    long_tracks_df.set_index(time_cols).index.isin(dup_idx)
].index
original_df_shape = normalized_df.shape
normalized_df = normalized_df.drop(index=to_drop_idx)
nor


# In[8]:


normalized_df.groupby(["Metadata_Well", "Metadata_FOV", "Metadata_track_id"]).size() > 1


# In[ ]:
