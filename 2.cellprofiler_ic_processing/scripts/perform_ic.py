#!/usr/bin/env python
# coding: utf-8

# # Create and edit LoadData csv and run CellProfiler `illum.cppipe` (IC) pipeline
#
# In this notebook, we run the CellProfiler IC pipeline to calculate the illumination (illum) correction functions for all images per channel (4).

# ## Import libraries

# In[ ]:


import pathlib
import sys

sys.path.append("../../utils")
import cp_utils as cp_utils
import tqdm

# ## Set paths

# In[ ]:


# path to input directory
path_to_input = pathlib.Path("../../data/20231004ChromaLive6hr_4ch_MaxIP").resolve()
# set paths for CellProfiler
path_to_pipeline = pathlib.Path("../pipelines/illum.cppipe").resolve()
## path to folder for IC functions
illum_directory = pathlib.Path("../illum_directory").resolve()


# ## Define the input paths

# In[ ]:


dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "path_to_input": pathlib.Path(
            "../../data/20230920ChromaLiveTL_24hr4ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20230920ChromaLiveTL_24hr4ch_MaxIP/"
        ).resolve(),
    },
    "run_20231004ChromaLive6hr_4ch_MaxIP": {
        "path_to_input": pathlib.Path(
            "../../data/20231004ChromaLive6hr_4ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231004ChromaLive6hr_4ch_MaxIP/"
        ).resolve(),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_input": pathlib.Path(
            "../../data/20231017ChromaLive_6hr_4ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "path_to_input": pathlib.Path(
            "../../data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{illum_directory}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/"
        ).resolve(),
    },
}


# ## Run `illum.cppipe` pipeline and calculate + save IC functions

# In[ ]:


for run in tqdm.tqdm(dict_of_inputs):
    path_to_input = dict_of_inputs[run]["path_to_input"]
    path_to_output = dict_of_inputs[run]["path_to_output"]
    # Run CellProfiler on the illum pipeline
    cp_utils.run_cellprofiler(
        path_to_pipeline=path_to_pipeline,
        path_to_input=path_to_input,
        path_to_output=path_to_output,
    )
