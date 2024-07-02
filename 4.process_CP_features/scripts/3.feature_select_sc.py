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


# directory where normalized parquet file is located
data_dir = pathlib.Path("../data/normalized_data")

# directory where the feature selected parquet file is saved to
output_dir = pathlib.Path("../data/feature_selected_data")
output_dir.mkdir(exist_ok=True)


# ## Define dict of paths

# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "normalized_df_path": pathlib.Path(
            f"{data_dir}/run_20230920ChromaLiveTL_24hr4ch_MaxIP_norm.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20230920ChromaLiveTL_24hr4ch_MaxIP_norm_fs.parquet"
        ).resolve(),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "normalized_df_path": pathlib.Path(
            f"{data_dir}/run_20231017ChromaLive_6hr_4ch_MaxIP_norm.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20231017ChromaLive_6hr_4ch_MaxIP_norm_fs.parquet"
        ).resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "normalized_df_path": pathlib.Path(
            f"{data_dir}/run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_norm.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_norm_fs.parquet"
        ).resolve(),
    },
}


# ## Perform feature selection

# In[4]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]


# In[5]:


manual_block_list = [
    "Nuclei_TrackObjects_Displacement_50",
    "Nuclei_TrackObjects_DistanceTraveled_50",
    "Nuclei_TrackObjects_IntegratedDistance_50",
    "Nuclei_TrackObjects_Label_50",
    "Nuclei_TrackObjects_Linearity_50",
    "Nuclei_TrackObjects_ParentObjectNumber_50",
    "Nuclei_AreaShape_BoundingBoxArea",
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Cells_AreaShape_BoundingBoxArea",
]


# In[6]:


# feature selection parameters
print(f"Performing feature selection on normalized annotated merged single cells!")
for info, input_path in dict_of_inputs.items():
    # read in the annotated file
    normalized_df = pd.read_parquet(input_path["normalized_df_path"])
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
        output_filename=input_path["outoput_file_path"],
        output_type="parquet",
    )
    print(
        f"Features have been selected for PBMC cells and saved to {pathlib.Path(info).name}!"
    )
    # check to see if the shape of the df has changed indicating feature selection occurred
    print(feature_select_df.shape)
    feature_select_df.head()
