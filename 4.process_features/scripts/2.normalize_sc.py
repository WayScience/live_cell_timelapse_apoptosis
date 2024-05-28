#!/usr/bin/env python
# coding: utf-8

# # Normalize annotated single cells using negative control (DSMO 0.025% and DMSO 0.100%)

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
from pycytominer import normalize
from pycytominer.cyto_utils import output

# ## Set paths and variables

# In[2]:


# directory where combined parquet file are located
data_dir = pathlib.Path("../data/annotated_data")

# directory where the normalized parquet file is saved to
output_dir = pathlib.Path("../data/normalized_data")
output_dir.mkdir(exist_ok=True)


# ## Define dict of paths

# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "annotated_file_path": pathlib.Path(
            f"{data_dir}/run_20230920ChromaLiveTL_24hr4ch_MaxIP_sc.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20230920ChromaLiveTL_24hr4ch_MaxIP_norm.parquet"
        ).resolve(),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "annotated_file_path": pathlib.Path(
            f"{data_dir}/run_20231017ChromaLive_6hr_4ch_MaxIP_sc.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20231017ChromaLive_6hr_4ch_MaxIP_norm.parquet"
        ).resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "annotated_file_path": pathlib.Path(
            f"{data_dir}/run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_sc.parquet"
        ).resolve(),
        "outoput_file_path": pathlib.Path(
            f"{output_dir}/run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_norm.parquet"
        ).resolve(),
    },
}


# ## Normalize with standardize method with negative control on annotated data

# In[4]:


for info, input_path in dict_of_inputs.items():
    # read in the annotated file
    print(input_path)
    annotated_df = pd.read_parquet(input_path["annotated_file_path"])

    # normalize annotated data
    normalized_df = normalize(
        # df with annotated raw merged single cell features
        profiles=annotated_df,
        # specify samples used as normalization reference (negative control)
        samples="Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0 and Image_Metadata_Time == '0001'",
        # normalization method used
        method="standardize",
    )
    output(
        normalized_df,
        output_filename=input_path["outoput_file_path"],
        output_type="parquet",
    )
    print(
        f"Single cells have been normalized for PBMC cells and saved to {pathlib.Path(info).name} !"
    )
    # check to see if the features have been normalized
    print(normalized_df.shape)
    normalized_df.head()
