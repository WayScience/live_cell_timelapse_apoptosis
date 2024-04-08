#!/usr/bin/env python
# coding: utf-8

# This notebooks generates platemaps for annotating the data with the Metadata

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# ## Define paths for metadata and platemap outputs

# In[2]:


# load in path
metadata_24h_path = pathlib.Path("../../data/metadata_24h.csv").resolve(strict=True)
metadata_6h_4ch_path = pathlib.Path("../../data/metadata_6hr_4ch.csv").resolve(
    strict=True
)
metadata_6h_2ch_path = pathlib.Path("../../data/metadata_AnnexinV_2ch.csv").resolve(
    strict=True
)

# output platemap paths
platemap_24h_path = pathlib.Path("../../data/platemap_24h.csv").resolve()
platemap_6h_4ch_path = pathlib.Path("../../data/platemap_6hr_4ch.csv").resolve()
platemap_6h_2ch_path = pathlib.Path("../../data/platemap_AnnexinV_2ch.csv").resolve()


# ## 24Hr - 4 channel data

# In[3]:


# load in metadata
metadata_24h = pd.read_csv(metadata_24h_path)  # remove the columns that are not needed
columns_to_remove = [
    "filename",
    "site",
    "channel",
    "timepoint",
    "zslice",
    "subdir_folder",
]
metadata_24h = metadata_24h.drop(columns=columns_to_remove)
metadata_24h
# replace NaN with test
metadata_24h = metadata_24h.fillna("test")

# drop duplicates
metadata_24h = metadata_24h.drop_duplicates()
metadata_24h.reset_index(drop=True, inplace=True)
# write to platemap
metadata_24h.to_csv(platemap_24h_path, index=False)

metadata_24h.head()


# ## 6Hr - 4 channel data

# In[4]:


# load in metadata
metadata_6h = pd.read_csv(
    metadata_6h_4ch_path
)  # remove the columns that are not needed
columns_to_remove = [
    "filename",
    "site",
    "channel",
    "timepoint",
    "zslice",
    "subdir_folder",
]
metadata_6h = metadata_6h.drop(columns=columns_to_remove)
metadata_6h
# replace NaN with test
metadata_6h = metadata_6h.fillna("test")
# drop duplicates
metadata_6h = metadata_6h.drop_duplicates()
metadata_6h.reset_index(drop=True, inplace=True)
# write to platemap
metadata_6h.to_csv(platemap_6h_4ch_path, index=False)
metadata_6h.head()


# ## 24Hr - 2 channel data

# In[5]:


# load in metadata
metadata_6h_2ch = pd.read_csv(
    metadata_6h_2ch_path
)  # remove the columns that are not needed
columns_to_remove = [
    "filename",
    "site",
    "channel",
    "timepoint",
    "zslice",
    "subdir_folder",
]
metadata_6h_2ch = metadata_6h_2ch.drop(columns=columns_to_remove)
metadata_6h_2ch
# replace NaN with test
metadata_6h_2ch = metadata_6h_2ch.fillna("test")
# write to platemap
metadata_24h.to_csv(platemap_6h_2ch_path, index=False)
metadata_6h_2ch.head()
