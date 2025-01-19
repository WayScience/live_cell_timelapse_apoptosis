#!/usr/bin/env python
# coding: utf-8

# This notebook converts the terminal time point data into the correct sequence time point for the main dataset.
# This will allow for the relation of the nuclei tracked in time to the main dataset.
# This way, the terminal stain can be related to the main (13 time points) dataset.

# There are four channels in the main dataset:
# Channel 1: DAPI
# Channel 2: CL488-1
# Channel 3: CL488-2
# Channel 4: CL561
# 
# There are two channels in the terminal dataset:
# Channel 1: DAPI
# Channel 5: Annexin V
# 
# Note that Channel 5 does not exists in the first 13 time points only the terminal timepoints. 
# Similarly, the terminal time points do not have the CL488-1, CL488-2, and CL561 channels.

# In[1]:


import argparse
import glob
import pathlib
import sys

import numpy as np
import pandas as pd


# In[2]:


illum_directory = pathlib.Path("../illum_directory").resolve(strict=True)
# get all directories in the illum_directory recursively
illum_directories = glob.glob(str(illum_directory) + "/**/", recursive=True)
# get all files in the illum_directories
illum_files = [glob.glob(directory + "/*") for directory in illum_directories]
# filter for files
illum_files = [
    file for sublist in illum_files for file in sublist if pathlib.Path(file).is_file()
]


# In[3]:


for file in illum_files:
    if "Annexin" in file:
        if "T0001" in file:
            file = pathlib.Path(file)
            file.rename(file.with_name(file.name.replace("T0001", "T0014")))
            print(file)

