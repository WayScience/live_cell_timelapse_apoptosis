#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import lance
import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from tqdm import tqdm

# In[2]:


# create the database object
uri = pathlib.Path("../../../data/objects_db").resolve()
# delete the database directory if it exists
if uri.exists():
    os.system(f"rm -rf {uri}")
db = lancedb.connect(uri)


# In[3]:


# set the path to the videos
tiff_dir = pathlib.Path(
    "../../../2.cellprofiler_ic_processing/illum_directory_test/20231017ChromaLive_6hr_4ch_MaxIP/"
).resolve(strict=True)


# ### Get data formatted correctly

# In[4]:


# get the list of tiff files in the directory
tiff_files = list(tiff_dir.glob("*.tiff"))
tiff_file_names = [file.stem for file in tiff_files]
# files to df
tiff_df = pd.DataFrame({"file_name": tiff_file_names, "file_path": tiff_files})

# split the file_path column by _ but keep the original column
tiff_df["file_name"] = tiff_df["file_name"].astype(str)
tiff_df[["Well", "FOV", "Timepoint", "Z-slice", "Channel", "illum"]] = tiff_df[
    "file_name"
].str.split("_", expand=True)
tiff_df["Well_FOV"] = tiff_df["Well"] + "_" + tiff_df["FOV"]
# drop all channels except for the first one
tiff_df = tiff_df[tiff_df["Channel"] == "C01"]
tiff_df = tiff_df.drop(columns=["Channel", "illum"])

# cast all types to string
tiff_df = tiff_df.astype(str)
# load binary data into the df of each image
tiff_df["image"] = tiff_df["file_path"].apply(lambda x: tifffile.imread(x).flatten())
tiff_df["binary_image"] = tiff_df["image"].apply(lambda x: x.tobytes())
# sort the df by the well, fov, timepoint, z-slice
tiff_df = tiff_df.sort_values(["Well", "FOV", "Timepoint", "Z-slice"])
tiff_df.reset_index(drop=True, inplace=True)
tiff_df.head()


# In[5]:


schema = pa.schema(
    [
        pa.field("file_name", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("Well", pa.string()),
        pa.field("FOV", pa.string()),
        pa.field("Timepoint", pa.string()),
        pa.field("Z-slice", pa.string()),
        pa.field("Well_FOV", pa.string()),
        pa.field("image", pa.list_(pa.int16())),
        # add binary data
        pa.field("binary_image", pa.binary()),
    ]
)
tbl = db.create_table("0.original_images", schema=schema, mode="overwrite")

tbl.add(tiff_df)


# Check to ensure the df is retrievable and formatted correctly

# In[6]:


df = tbl.to_pandas()
df.head()


# In[7]:


# load the first image
df["image"][0]
# load the binary data into a numpy array
np.frombuffer(df["image"][0], dtype=np.uint8)
# plto the image

df["image"][0].reshape(1900, 1900)
plt.imshow(df["image"][0].reshape(1900, 1900), cmap="gray")
plt.show()
