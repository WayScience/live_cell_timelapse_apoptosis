#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
from pycytominer import annotate
from pycytominer.cyto_utils import output

# ## Set paths and variables

# In[2]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# directory where parquet files are located
data_dir = pathlib.Path("../data/converted_data")

# directory where the annotated parquet files are saved to
output_dir = pathlib.Path("../data/annotated_data")
output_dir.mkdir(exist_ok=True)


# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/20230920ChromaLiveTL_24hr4ch_MaxIP.parquet"
        ).resolve(strict=True),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_24h.csv").resolve(
            strict=True
        ),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/20231017ChromaLive_6hr_4ch_MaxIP.parquet"
        ).resolve(strict=True),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_6hr_4ch.csv").resolve(
            strict=True
        ),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP.parquet"
        ).resolve(strict=True),
        "platemap_path": pathlib.Path(
            f"{platemap_path}/platemap_AnnexinV_2ch.csv"
        ).resolve(strict=True),
    },
}


# ## Annotate merged single cells

# In[4]:


for data_run, info in dict_of_inputs.items():
    # load in converted parquet file as df to use in annotate function
    single_cell_df = pd.read_parquet(info["source_path"])
    single_cell_df.rename(
        columns={
            "Image_Metadata_FOV": "Metadata_FOV",
            "Image_Metadata_Time": "Metadata_Time",
        },
        inplace=True,
    )
    platemap_df = pd.read_csv(info["platemap_path"])
    output_file = str(pathlib.Path(f"{output_dir}/{data_run}_sc.parquet"))
    print(f"Adding annotations to merged single cells for {data_run}!")

    # add metadata from platemap file to extracted single cell features
    annotated_df = annotate(
        profiles=single_cell_df,
        platemap=platemap_df,
        join_on=["Metadata_well", "Image_Metadata_Well"],
    )

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

    # save annotated df as parquet file
    output(
        df=annotated_df,
        output_filename=output_file,
        output_type="parquet",
    )
    print(f"Annotations have been added to {data_run} and saved!")
    # check last annotated df to see if it has been annotated correctly
    print(annotated_df.shape)
    annotated_df.head()
