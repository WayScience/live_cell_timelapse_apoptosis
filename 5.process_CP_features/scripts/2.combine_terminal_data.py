#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import lancedb
import pandas as pd
from pycytominer.cyto_utils import output

# In[2]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# directory where the annotated parquet files are saved to
input_dir = pathlib.Path("../data/annotated_data")
input_dir.mkdir(exist_ok=True)

# directory for the output combined files
output_dir = pathlib.Path("../data/combined_terminal_data")
output_dir.mkdir(exist_ok=True)


# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{input_dir}/run_20231017ChromaLive_6hr_4ch_MaxIP_sc.parquet"
        ).resolve(strict=True),
        # same file name but different path
        "output_path": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_6hr_4ch_MaxIP_sc.parquet"
        ).resolve(),
    },
    "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{input_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_sc.parquet"
        ).resolve(strict=True),
        "output_path": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_sc.parquet"
        ).resolve(),
    },
}


# ### load the database into the memory

# In[4]:


# set and connect to the db
# create the database object
uri = pathlib.Path("../../data/objects_db").resolve()
db = lancedb.connect(uri)
# get the db schema and tables
db.table_names()
# load table
table = db["1.masked_images"]
location_metadata_df = table.to_pandas()
print(location_metadata_df.shape)
location_metadata_df.head()
# change frame to Metadata_Time
location_metadata_df.rename(columns={"frame": "Metadata_Time"}, inplace=True)
# add 1 to Metadata_Time to match the timepoints in the single cell data
location_metadata_df["Metadata_Time"] = location_metadata_df["Metadata_Time"] + 1
# change formatting to leading 4 zeros
location_metadata_df["Metadata_Time"] = location_metadata_df["Metadata_Time"].apply(
    lambda x: f"{x:04}"
)
print(location_metadata_df.shape)
location_metadata_df.head()


# In[5]:


location_metadata_df["Metadata_Time"].unique()


# In[6]:


# split the dataframes by terminal time and non-terminal time
terminal_location_metadata_df = location_metadata_df.loc[
    location_metadata_df["Metadata_Time"] == "0014"
]
print(terminal_location_metadata_df.shape)


# ### Merge the terminal and single cell data

# In[7]:


main_df = pd.read_parquet(
    dict_of_inputs["run_20231017ChromaLive_6hr_4ch_MaxIP"]["source_path"]
)
terminal_df = pd.read_parquet(
    dict_of_inputs["20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"]["source_path"]
)

print(main_df.shape)
print(terminal_df.shape)


# In[12]:


main_df.head()
main_df["Metadata_object_id"].unique()


# In[8]:


terminal_df.head()


# In[ ]:


# In[ ]:
