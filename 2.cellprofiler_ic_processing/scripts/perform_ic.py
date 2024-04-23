#!/usr/bin/env python
# coding: utf-8

# # Run CellProfiler `illum.cppipe` (IC) pipeline
# 
# In this notebook, we run the CellProfiler IC pipeline to calculate the illumination (illum) correction functions for all images per channel (4), apply the functions, and save images into a new directory.

# ## Import libraries

# In[ ]:


import pathlib
import sys

sys.path.append("../../utils")
import cp_parallel
import cp_utils as cp_utils
import tqdm


# ## Set paths

# In[ ]:


run_name = "illumination_correction"
# path to folder for IC functions
illum_directory = pathlib.Path("../illum_directory").resolve()
# make sure the directory exists
illum_directory.mkdir(exist_ok=True, parents=True)


# ## Define the input paths

# In[ ]:


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
}


# ## Run `illum.cppipe` pipeline and calculate + save IC functions

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_inputs, run_name=run_name
)

