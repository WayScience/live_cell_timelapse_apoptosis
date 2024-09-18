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


run_name = "illumination_correction"
# path to folder for IC images
illum_directory = pathlib.Path("../illum_directory").resolve()
# make sure the directory exists
illum_directory.mkdir(exist_ok=True, parents=True)


# ## Define the input paths

# In[3]:


dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            "../../data/20230920ChromaLiveTL_24hr4ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20230920ChromaLiveTL_24hr4ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/illum_4ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            "../../data/20231017ChromaLive_6hr_4ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/illum_4ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "path_to_images": pathlib.Path(
            "../../data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/illum_2ch.cppipe").resolve(),
    },
    # testing small datasets to make sure the pipeline works
    # these have both Well C02 FOV 1 and Well E11 FOV 4
    "run_20231017ChromaLive_6hr_4ch_MaxIP_test_small": {
        "path_to_images": pathlib.Path(
            "../../data/20231017ChromaLive_6hr_4ch_MaxIP_test_small"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_6hr_4ch_MaxIP_test_small/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/illum_4ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small": {
        "path_to_images": pathlib.Path(
            "../../data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/illum_2ch.cppipe").resolve(),
    },
}


# ## Run `illum.cppipe` pipeline and calculate + save IC images
# This last cell does not get run as we run this pipeline in the command line.

# In[4]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_inputs, run_name=run_name
)

