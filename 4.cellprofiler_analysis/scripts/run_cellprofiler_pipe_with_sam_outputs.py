#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pprint
import shutil
import sys

sys.path.append("../../utils/")
import cp_parallel


# ## Set paths and variables

# In[ ]:


HPC = False


# In[2]:


# set the run type for the parallelization
run_name = "analysis"

# set main output dir for all plates
output_dir = pathlib.Path("../analysis_output")
output_dir.mkdir(exist_ok=True, parents=True)

# directory where images are located within folders
images_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_6hr_4ch_MaxIP_test_small"
).resolve()
# directory where masks are located within folders
masks_dir = pathlib.Path("../../3b.run_sam/sam2_processing_dir/masks").resolve()

# path to plugins directory as one of the pipelines uses the RunCellpose plugin
plugins_dir = pathlib.Path(
    "/home/lippincm/Documents/CellProfiler-plugins/active_plugins"
).resolve()

HPC_plugins_dir = pathlib.Path("/scratch/alpine/mlippincott@xsede.org/").resolve()


# In[3]:


# make a new dir for input images
CP_input_dir = pathlib.Path("../../3b.run_sam/sam2_processing_dir/CP_input/").resolve()
# remove any existing files in the dir from previous runs
if CP_input_dir.exists():
    shutil.rmtree(CP_input_dir)
CP_input_dir.mkdir(exist_ok=True, parents=True)

# copy all images to the new dir
for image in images_dir.rglob("*.tiff"):
    if image.is_file():
        shutil.copy(image, CP_input_dir)
for mask in masks_dir.rglob("*.png"):
    if mask.is_file():
        # check if the mask is a terminal mask
        if not "T0014" in mask.stem:
            shutil.copy(mask, CP_input_dir)


# ## Create dictionary with all info for each plate

# In[4]:


dict_of_inputs = {
    "20231017ChromaLive_6hr_4ch_MaxIP_sam": {
        "path_to_images": pathlib.Path(f"{CP_input_dir}").resolve(),
        "path_to_output": pathlib.Path(
            f"{output_dir}/20231017ChromaLive_6hr_4ch_MaxIP_test_small/"
        ).resolve(),
        "path_to_pipeline": pathlib.Path(
            "../pipelines/analysis_4ch_with_sam.cppipe"
        ).resolve(),
    },
    # "run_20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
    #     "path_to_images": pathlib.Path(
    #         "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small"
    #     ).resolve(strict=True),
    #     "path_to_output": pathlib.Path(
    #         f"{output_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small/"
    #     ).resolve(),
    #     "path_to_pipeline": pathlib.Path("../pipelines/analysis_2ch.cppipe").resolve(),
    # },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(dict_of_inputs, indent=4)


# In[5]:


if not HPC:
    cp_parallel.run_cellprofiler_parallel(
        plate_info_dictionary=dict_of_inputs,
        run_name=run_name,
        plugins_dir=plugins_dir,
    )
else:
    cp_parallel.run_cellprofiler_parallel(
        plate_info_dictionary=dict_of_inputs,
        run_name=run_name,
        plugins_dir=HPC_plugins_dir,
    )

