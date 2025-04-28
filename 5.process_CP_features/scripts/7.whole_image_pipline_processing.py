#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pathlib
import sqlite3

import pandas as pd
from pycytominer import aggregate, annotate, feature_select, normalize

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
aggregated_data_dir = pathlib.Path(
    output_dir, "aggregated_whole_image.parquet"
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


# In[ ]:


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

list_of_col_to_remove = []
for col in df.columns:
    for keyword in blacklist_keywords:
        if keyword in col:
            list_of_col_to_remove.append(col)
df.drop(columns=list_of_col_to_remove, inplace=True)

for col in df.columns:
    if col.startswith("Image_"):
        df.rename(columns={col: col.replace("Image_", "")}, inplace=True)
print(df.shape)


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
print(annotated_df.shape)
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
normalized_df = normalized_df.drop_duplicates()
normalized_df = normalized_df.reset_index(drop=True)
print(normalized_df.shape)
normalized_df.to_parquet(normalized_data_dir, index=False)


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
feature_select_df.to_parquet(
    feature_selected_data_dir,
    index=False,
)
print(feature_select_df.shape)
feature_select_df.head()


# ## Aggregation

# In[ ]:


metadata_cols = feature_select_df.columns[
    feature_select_df.columns.str.contains("Metadata")
]
feature_cols = feature_select_df.columns[
    ~feature_select_df.columns.str.contains("Metadata")
].to_list()

aggregated_df = aggregate(
    feature_select_df,
    features=feature_cols,
    strata=["Metadata_Well", "Metadata_dose"],
    operation="median",
)

print(aggregated_df.shape)
aggregated_df.to_parquet(aggregated_data_dir)
print(aggregated_df.shape)
aggregated_df.head()
