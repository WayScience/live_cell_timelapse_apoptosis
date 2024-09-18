#!/usr/bin/env python
# coding: utf-8

# This notebook converts the terminal time point data into the correct sequence time point for the main dataset.
# This will allow for the relation of the nuceli tracked in time to the main dataset.
# This way, the terminal stain can be related to the main dataset.

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


import pathlib

import numpy as np
import pandas as pd

# In[2]:


# set the path to terminal data
terminal_data_path = pathlib.Path(
    "../illum_directory/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small"
).resolve(strict=True)

# number of timepoints in the main data
num_timepoints = 13


# get the list of files in the terminal data directory that are tiffs
tiff_files = list(terminal_data_path.glob("*.tiff"))
tiff_files = sorted(tiff_files)
# change the timepoint from "T0001" to "T0014" to match the main data format and position
# tiff_files = [str(file).replace("T0001", "T0014") for file in tiff_files]
# rewrite the list of files to the terminal data directory
for f in tiff_files:
    print(f)
    print(f.with_name(f.name.replace("T0001", "T0014")))
    f.rename(f.with_name(f.name.replace("T0001", "T0014")))
