#!/usr/bin/env python
# coding: utf-8

# # Run CellProfiler `illum.cppipe` (IC) pipeline
#
# In this notebook, we run the CellProfiler IC pipeline to calculate the illumination (illum) correction functions for all images per channel (4), apply the functions, and save images into a new directory.

# ## Import libraries

# In[1]:


import pathlib
import sys

sys.path.append("../../utils")
import cp_parallel
import cp_utils as cp_utils
import tqdm

# ## Set paths

# In[2]:


run_test_data = True

if not run_test_data:
    preprocessed_data_path = pathlib.Path("../../data/preprocessed_data/").resolve()
    illum_directory = pathlib.Path("../illum_directory").resolve()

else:
    preprocessed_data_path = pathlib.Path("../../data/test_data/").resolve()
    illum_directory = pathlib.Path("../illum_directory_test").resolve()


illum_directory.mkdir(exist_ok=True, parents=True)


# ## Define the input paths

# In[3]:


dict_of_inputs = {}


# In[4]:


# get the list of dirs in the raw_data_path
dirs = [x for x in preprocessed_data_path.iterdir() if x.is_dir()]
# get the list of all dirs in the dir
for dir in dirs:
    # get the list of all dirs in the dir
    subdirs = [x for x in dir.iterdir() if x.is_dir()]
    for subdir in subdirs:
        run_name = f"{dir.name}_{subdir.name}"
        if "4ch" in dir.name:
            dict_of_inputs[run_name] = {
                "path_to_images": pathlib.Path(subdir).resolve(strict=True),
                "path_to_output": pathlib.Path(illum_directory / run_name).resolve(),
                "path_to_pipeline": pathlib.Path(
                    "../pipelines/illum_4ch.cppipe"
                ).resolve(),
            }
        elif "2ch" in dir.name:
            dict_of_inputs[run_name] = {
                "path_to_images": pathlib.Path(subdir).resolve(strict=True),
                "path_to_output": pathlib.Path(illum_directory / run_name).resolve(),
                "path_to_pipeline": pathlib.Path(
                    "../pipelines/illum_2ch.cppipe"
                ).resolve(),
            }
        else:
            ValueError("The directory name does not contain 2ch or 4ch")


# ## Run `illum.cppipe` pipeline and calculate + save IC images
# This last cell does not get run as we run this pipeline in the command line.

# In[5]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_inputs, run_name=run_name
)
