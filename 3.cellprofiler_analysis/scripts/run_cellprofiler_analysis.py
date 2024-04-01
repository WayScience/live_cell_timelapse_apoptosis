#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[ ]:


import pathlib
import pprint
import sys

sys.path.append("../../utils/")
import cp_parallel

# ## Set paths and variables

# In[ ]:


# set the run type for the parallelization
run_name = "analysis"

# set main output dir for all plates
output_dir = pathlib.Path("../analysis_output")
output_dir.mkdir(exist_ok=True, parents=True)

# directory where images are located within folders
images_dir = pathlib.Path("../../2.cellprofiler_ic_processing/illum_directory")


# ## Create dictionary with all info for each plate

# In[ ]:


dict_of_inputs = {
    "run_20230920ChromaLiveTL_24hr4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20230920ChromaLiveTL_24hr4ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20230920ChromaLiveTL_24hr4ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/analysis_4ch.cppipe").resolve(),
    },
    "run_20231004ChromaLive6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231004ChromaLive6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20231004ChromaLive6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/analysis_4ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/analysis_4ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path("../pipelines/analysis_2ch.cppipe").resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_whole_image": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_whole_image/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path(
            "../pipelines/analysis_2ch_image.cppipe"
        ).resolve(),
    },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(dict_of_inputs, indent=4)


# ## Run analysis pipeline on each plate in parallel
#
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_inputs, run_name=run_name
)
