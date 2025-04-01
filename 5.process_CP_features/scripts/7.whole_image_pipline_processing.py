#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import pathlib
import sqlite3

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cytotable import convert, presets
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from pycytominer import annotate, feature_select, normalize
from pycytominer.cyto_utils import output

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[2]:


sqlite_path = pathlib.Path(
    "../../4.cellprofiler_analysis/analysis_output/endpoint_whole_image"
).resolve()
# get the files in the children directories
sqlite_files = glob.glob(f"{sqlite_path}/**/*.sqlite", recursive=True)


# In[3]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/platemap_AnnexinV_2ch.csv").resolve()
platemap_df = pd.read_csv(platemap_path)

# directory where the annotated parquet files are saved to
output_dir = pathlib.Path("../data/endpoint_whole_image/").resolve()
output_dir.mkdir(exist_ok=True, parents=True)

normalized_data_dir = pathlib.Path(
    output_dir, "normalized_whole_image.parquet"
).resolve()
feature_selected_data_dir = pathlib.Path(
    output_dir, "feature_selected_whole_image.parquet"
).resolve()


# ## Convert

# In[4]:


preset = """SELECT * FROM Per_Image;"""


# In[5]:


blacklist_keywords = [
    "Skeleton",
    "URL",
    "ExecutionTime",
    "Frame",
    "Group",
    "Height",
    "Width",
    "MD5",
    "Scaling",
    "Series",
]


# In[6]:


list_of_dfs = []
for file in sqlite_files:
    source_path = pathlib.Path(file)
    output_file_dir = output_dir / source_path.stem
    # get the path to the sqlite file
    with sqlite3.connect(source_path) as conn:
        query = "SELECT * FROM Per_Image;"
        df = pd.read_sql_query(query, conn)
    list_of_dfs.append(df)

df = pd.concat(list_of_dfs, ignore_index=True)
df = df.drop_duplicates()
# Save the DataFrame to a Parquet file

# df.to_parquet(output_parquet_path, index=False)
list_of_col_to_remove = []
for col in df.columns:
    for keyword in blacklist_keywords:
        if keyword in col:
            list_of_col_to_remove.append(col)
df.drop(columns=list_of_col_to_remove, inplace=True)

for col in df.columns:
    if col.startswith("Image_"):
        df.rename(columns={col: col.replace("Image_", "")}, inplace=True)


# ## Annotate

# In[7]:


# add metadata from platemap file to extracted single cell features
annotated_df = annotate(
    profiles=df,
    platemap=platemap_df,
    join_on=["Metadata_well", "Metadata_Well"],
)
# drop duplicate columns
annotated_df.drop_duplicates(inplace=True)
columns_to_drop = [
    "ImageNumber",
    "FileName_AnnexinV",
    "FileName_DNA",
    "PathName_AnnexinV",
    "PathName_DNA",
]
annotated_df.drop(columns=columns_to_drop, inplace=True)
annotated_df.head()


# ## Normalize

# In[8]:


metadata_columns = [x for x in annotated_df.columns if "Metadata_" in x]
feature_columns = [x for x in annotated_df.columns if "Metadata_" not in x]


# In[9]:


normalized_df = normalize(
    # df with annotated raw merged single cell features
    profiles=annotated_df,
    # specify samples used as normalization reference (negative control)
    samples="Metadata_compound == 'Staurosporine' and Metadata_dose == 0.0",
    # normalization method used
    method="standardize",
    features=feature_columns,
    meta_features=metadata_columns,
)


# ## Feature selection

# In[10]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]

feature_select_df = feature_select(
    normalized_df,
    operation=feature_select_ops,
    # specify features to be used for feature selection
    features=feature_columns,
)


print(f"Number of features before feature selection: {normalized_df.shape[1]}")
print(f"Number of features after feature selection: {feature_select_df.shape[1]}")
feature_select_df.head()
