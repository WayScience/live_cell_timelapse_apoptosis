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

# In[ ]:


import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[ ]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--test_data",
        type=bool,
        default=False,
        help="Use test data",
    )

    args = parser.parse_args()
    run_test_data = args.test_data
else:
    print("Running in a notebook")
    run_test_data = True


if not run_test_data:
    illum_directory = pathlib.Path("../illum_directory").resolve()

else:
    illum_directory = pathlib.Path("../illum_directory_test").resolve()

experiment = "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"


# In[3]:


# get the list of dirs in the raw_data_path
dirs = [x for x in illum_directory.iterdir() if x.is_dir()]
# get the list of all dirs in the dir
for dir in dirs:
    if experiment in dir.name:
        for image in dir.glob("*.tiff"):
            if "T0001" in image.name:
                image.rename(image.with_name(image.name.replace("T0001", "T0014")))

