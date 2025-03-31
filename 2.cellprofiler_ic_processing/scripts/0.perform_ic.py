#!/usr/bin/env python
# coding: utf-8

# # Run CellProfiler `illum.cppipe` (IC) pipeline
# 
# In this notebook, we run the CellProfiler IC pipeline to calculate the illumination (illum) correction functions for all images per channel (4), apply the functions, and save images into a new directory.

# ## Import libraries

# In[1]:


import argparse
import pathlib
import sys

sys.path.append("../../utils")
from cp_utils import run_cellprofiler


# In[7]:


# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing the images to be segmented",
        required=True,
    )
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve()
else:
    print("Running in a notebook")
    input_dir = pathlib.Path(
        "../../data/test_data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/C-02_F0001"
    ).resolve()


# ## Set paths

# In[12]:


if "test" in str(input_dir):
    illum_directory = pathlib.Path("../illum_directory/test_data/").resolve()
else:
    illum_directory = pathlib.Path("../illum_directory/").resolve()

if "Annexin" in str(input_dir):
    illum_directory = pathlib.Path(f"{illum_directory}/endpoint").resolve()
    path_to_pipeline = pathlib.Path("../pipelines/illum_2ch.cppipe").resolve()
else:
    illum_directory = pathlib.Path(f"{illum_directory}/timelapse").resolve()
    path_to_pipeline = pathlib.Path("../pipelines/illum_4ch.cppipe").resolve()

illum_directory.mkdir(parents=True, exist_ok=True)


# In[13]:


illum_name = str(input_dir).split("/")[-2] + "_" + str(input_dir).split("/")[-1]
print(illum_name)


# ## Define the input paths

# In[14]:


path_to_output = pathlib.Path(f"{illum_directory}/{illum_name}").resolve()


# ## Run `illum.cppipe` pipeline and calculate + save IC images
# This last cell does not get run as we run this pipeline in the command line.

# In[15]:


run_cellprofiler(
    path_to_pipeline=path_to_pipeline,
    path_to_input=input_dir,
    path_to_output=path_to_output,
    sqlite_name=illum_name,
    rename_sqlite_file_bool=True,
)

