#!/usr/bin/env python
# coding: utf-8

# This notebook will sort the image files by Well and FOV across time for more effiecient parallel processing.
# The sorted images will be saved in a new directory.

# In[1]:


import pathlib
import shutil

import numpy as np
import pandas as pd

# In[ ]:


raw_data_path = pathlib.Path("../../data/raw_data/").resolve(strict=True)
preprocessed_data_path = pathlib.Path("../../data/preprocessed_data/").resolve()
preprocessed_data_path.mkdir(parents=True, exist_ok=True)
# get the list of dirs in the raw_data_path
dirs = [x for x in raw_data_path.iterdir() if x.is_dir()]


# In[ ]:


files_dict = {}
for dir in dirs:
    files = [x for x in dir.iterdir() if x.is_file()]
    files_dict[dir.name] = files

output_dict = {
    "experiment": [],
    "file_path": [],
    "file": [],
}
# loop through each experiment and get the file paths
for experiment, files in files_dict.items():
    new_data_path = pathlib.Path(preprocessed_data_path / experiment)
    new_data_path.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.suffix == ".npz" and f.suffix == ".tif":
            output_dict["experiment"].append(experiment)
            output_dict["file_path"].append(f)
            output_dict["file"].append(f.name)


files_df = pd.DataFrame(output_dict)
# loop through each experiment and group the files
for experiment in files_df["experiment"].unique():
    tmp_df = files_df[files_df["experiment"] == experiment]
    tmp_df["group"] = tmp_df["file"].str.split("_T", expand=True)[0]
    for group in sorted(tmp_df["group"].unique()):
        file_sorting_df = tmp_df[tmp_df["group"] == group]
        new_group_path = pathlib.Path(
            preprocessed_data_path / experiment / group
        ).resolve()
        new_group_path.mkdir(parents=True, exist_ok=True)
        for i, row in file_sorting_df.iterrows():
            file_name = row["file"]
            old_file_path = row["file_path"]
            new_file_path = pathlib.Path(new_group_path / file_name).resolve()
            shutil.copy(old_file_path, new_file_path)
