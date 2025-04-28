#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[1]:


import argparse
import pathlib
import pprint
import sys
import time

sys.path.append("../../utils/")
import cp_utils

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[2]:



# directory where images are located within folders
images_dir = pathlib.Path("../../2.cellprofiler_ic_processing/illum_directory")
experiment_prefix_timelapse = "20231017ChromaLive_6hr_4ch_MaxIP_"
experiment_prefix_endpoint = "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_"


# In[3]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Illumination correction")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
    timelapse_dir = pathlib.Path(
        f"{images_dir}/timelapse/{experiment_prefix_timelapse}{well_fov}/"
    )
    endpoint_dir = pathlib.Path(
        f"{images_dir}/endpoint/{experiment_prefix_endpoint}{well_fov}/"
    )
else:
    print("Running in a notebook")
    well_fov = "E-07_F0001"
    timelapse_dir = pathlib.Path(
        f"{images_dir}/timelapse/{experiment_prefix_timelapse}{well_fov}/"
    )
    endpoint_dir = pathlib.Path(
        f"{images_dir}/endpoint/{experiment_prefix_endpoint}{well_fov}/"

    )

path_to_pipelines = pathlib.Path("../pipelines/").resolve(strict=True)

# set main output dir for all plates
output_dir = pathlib.Path("../analysis_output/")
output_dir.mkdir(exist_ok=True, parents=True)


# ## Create dictionary with all info for each plate

# In[4]:


dict_of_inputs = {
    "20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(timelapse_dir).resolve(strict=True),

        "path_to_output": pathlib.Path(f"{output_dir}/timelapse/{well_fov}").resolve(),
        "path_to_pipeline": pathlib.Path(
            f"{path_to_pipelines}/analysis_4ch.cppipe"
        ).resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "path_to_images": pathlib.Path(endpoint_dir).resolve(),

        "path_to_output": pathlib.Path(f"{output_dir}/endpoint/{well_fov}").resolve(),
        "path_to_pipeline": pathlib.Path(
            f"{path_to_pipelines}/analysis_2ch.cppipe"
        ).resolve(),
    },
    "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_whole_image": {
        "path_to_images": pathlib.Path(endpoint_dir).resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/endpoint_whole_image/{well_fov}"
        ).resolve(),
        "path_to_pipeline": pathlib.Path(
            f"{path_to_pipelines}/analysis_2ch_image.cppipe"
        ).resolve(),
    },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(dict_of_inputs, indent=4)


# ## Run analysis pipeline on each plate in parallel
#
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[5]:


start = time.time()


# In[6]:


for run in dict_of_inputs.keys():
    cp_utils.run_cellprofiler(
        path_to_pipeline=dict_of_inputs[run]["path_to_pipeline"],
        path_to_input=dict_of_inputs[run]["path_to_images"],
        path_to_output=dict_of_inputs[run]["path_to_output"],
    )


# In[7]:


end = time.time()
# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)
