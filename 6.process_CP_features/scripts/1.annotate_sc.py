#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import argparse
import pathlib

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycytominer import annotate
from pycytominer.cyto_utils import output

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[2]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# directory where parquet files are located
data_dir = pathlib.Path("../data/0.converted_data").resolve()

# directory where the annotated parquet files are saved to
output_dir = pathlib.Path("../data/1.annotated_data")
output_dir.mkdir(exist_ok=True)

if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Single cell extraction")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
else:
    print("Running in a notebook")
    well_fov = "C-02_F0001"


# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(f"{data_dir}/timelapse/{well_fov}.parquet").resolve(
            strict=True
        ),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_6hr_4ch.csv").resolve(
            strict=True
        ),
        "output_file": pathlib.Path(
            f"{output_dir}/timelapse/{well_fov}_sc.parquet"
        ).resolve(),
    },
    "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/endpoint/{well_fov}.parquet"
        ).resolve(),
        "platemap_path": pathlib.Path(
            f"{platemap_path}/platemap_AnnexinV_2ch.csv"
        ).resolve(strict=True),
        "output_file": pathlib.Path(
            f"{output_dir}/endpoint/{well_fov}_sc.parquet"
        ).resolve(),
    },
}


# ## Annotate merged single cells

# In[4]:


for data_run, info in dict_of_inputs.items():
    # load in converted parquet file as df to use in annotate function
    single_cell_df = pd.read_parquet(info["source_path"])
    print(single_cell_df.shape)
    single_cell_df = single_cell_df.rename(
        columns={
            "Image_Metadata_FOV": "Metadata_FOV",
            "Image_Metadata_Time": "Metadata_Time",
        },
    )
    platemap_df = pd.read_csv(info["platemap_path"])

    print(f"Adding annotations to merged single cells for {data_run}!")

    # add metadata from platemap file to extracted single cell features
    annotated_df = annotate(
        profiles=single_cell_df,
        platemap=platemap_df,
        join_on=["Metadata_well", "Image_Metadata_Well"],
    )
    print(annotated_df.shape)

    # move metadata well and single cell count to the front of the df (for easy visualization in python)
    well_column = annotated_df.pop("Metadata_Well")
    singlecell_column = annotated_df.pop("Metadata_number_of_singlecells")
    # insert the column as the second index column in the dataframe
    annotated_df.insert(1, "Metadata_Well", well_column)
    annotated_df.insert(2, "Metadata_number_of_singlecells", singlecell_column)

    # rename metadata columns to match the expected column names
    columns_to_rename = {
        "Nuclei_Location_Center_Y": "Metadata_Nuclei_Location_Center_Y",
        "Nuclei_Location_Center_X": "Metadata_Nuclei_Location_Center_X",
    }
    # Image_FileName cols
    for col in annotated_df.columns:
        if "Image_FileName" in col:
            columns_to_rename[col] = f"Metadata_{col}"
        elif "Image_PathName" in col:
            columns_to_rename[col] = f"Metadata_{col}"
        elif "TrackObjects" in col:
            columns_to_rename[col] = f"Metadata_{col}"
    # rename metadata columns
    annotated_df.rename(columns=columns_to_rename, inplace=True)

    info["output_file"].parent.mkdir(exist_ok=True, parents=True)

    print(
        f"Annotations have been added to {data_run} and saved to {info['output_file']}"
    )
    # check last annotated df to see if it has been annotated correctly
    annotated_df.drop_duplicates(inplace=True)
    # save annotated df as parquet file
    output(
        df=annotated_df,
        output_filename=info["output_file"],
        output_type="parquet",
    )
    print(annotated_df.shape)
    annotated_df.head()
