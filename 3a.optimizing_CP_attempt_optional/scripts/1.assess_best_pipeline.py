#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import pprint
import sys

import optuna
import pandas as pd
import torch

sys.path.append("../CellProfiler_optuna_utils/")
from optuna_profiling_utils import (
    adjust_cpipe_file_LAP,
    adjust_cpipe_file_overlap,
    adjust_cpipe_file_save_tracked_object_images,
    extract_pixel_number_for_overlap_tracking,
    extract_single_time_cell_tracking_entropy,
    harmonic_mean,
    loss_function_from_CP_features,
    loss_function_MSE,
    remove_trial_intermediate_files,
    retrieve_cell_count,
    run_CytoTable,
    run_pyctominer_annotation,
)

sys.path.append("../../utils/")
import cp_parallel
from cytotable import convert, presets
from pycytominer import annotate
from pycytominer.cyto_utils import output

sys.path.append("../../utils")
import sc_extraction_utils as sc_utils
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

# In[2]:


# set up an argument parser
parser = argparse.ArgumentParser(description="Run CellProfiler pipelines with Optuna.")

parser.add_argument(
    "--tracking_type",
    "-t",
    type=str,
    default="overlap",
    help="The type of tracking to use. Options are 'overlap' or 'LAP'.",
    required=False,
)

# get the arguments
args = parser.parse_args()

# set the tracking type
tracking_type = args.tracking_type


# In[3]:


# clear out old trials and studies
remove_trial_intermediate_files(
    output_dir=pathlib.Path("../analysis_output/"),
)

# remove cpipe files
remove_trial_intermediate_files(
    output_dir=pathlib.Path("../pipelines/generated_pipelines").resolve(),
)


# In[4]:


# set main output dir for all plates
output_dir = pathlib.Path("../analysis_output")
output_dir.mkdir(exist_ok=True, parents=True)

# directory where images are located within folders
images_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory_test_small"
).resolve(strict=True)

# directory where the pipeline is located
plugins_dir = pathlib.Path(
    "/home/lippincm/Documents/CellProfiler-plugins/active_plugins"
).resolve(strict=True)


# In[5]:


# define the cpipe file path
if tracking_type == "LAP":
    cpipe_file = pathlib.Path(
        "../pipelines/cell_tracking_optimization_LAP_best_trial.cppipe"
    ).resolve(strict=True)
elif tracking_type == "overlap":
    cpipe_file = pathlib.Path(
        "../pipelines/cell_tracking_optimization_overlap_best_trial.cppipe"
    ).resolve(strict=True)
    # get the pixel numberts
    max_pixels = extract_pixel_number_for_overlap_tracking(
        cpipe_file=cpipe_file,
    )
else:
    raise ValueError(f"Tracking type {tracking_type} not recognized.")

adjust_cpipe_file_save_tracked_object_images(cpipe_file)


# In[6]:


dict_of_inputs_for_cellprofiler = {
    "20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path(
            f"../analysis_output/{tracking_type}/"
        ).resolve(),
        "path_to_pipeline": f"{cpipe_file}",
    },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(dict_of_inputs_for_cellprofiler, indent=4)


# ### CytoTable paths and set up

# In[7]:


# run CytoTable analysis for merged data
# type of file output from CytoTable (currently only parquet)

# dictionary of inputs for CytoTable to pass to the function
dict_of_inputs_for_cytotable = {
    "run_20231004ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"../analysis_output/{tracking_type}/Track_Objects.sqlite"
        ).resolve(),
        "dest_path": pathlib.Path(
            f"../analysis_output/{tracking_type}/Track_Objects.parquet"
        ).resolve(),
        "preset": """
            SELECT
                *
            FROM
                read_parquet('per_image.parquet') as per_image
            INNER JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.Metadata_ImageNumber = per_image.Metadata_ImageNumber
            """,
    },
}


# ### PyCytominer paths and set up

# In[8]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# dictionary with each run for the cell type
dict_of_inputs_for_pycytominer = {
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"../analysis_output/{tracking_type}/Track_Objects.parquet"
        ).resolve(),
        "output_file_path": pathlib.Path(
            f"../analysis_output/{tracking_type}/Track_Objects_sc.parquet"
        ).resolve(),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_6hr_4ch.csv").resolve(
            strict=True
        ),
    }
}


# In[9]:


# run cellprofiler pipeline
cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_inputs_for_cellprofiler,
    run_name="testing_params",
    plugins_dir=plugins_dir,
)

# run CytoTable analysis for merged data
run_CytoTable(cytotable_dict=dict_of_inputs_for_cytotable)

# run the annotation function
output_file_path = run_pyctominer_annotation(
    pycytominer_dict=dict_of_inputs_for_pycytominer,
)

if tracking_type == "overlap":
    _columns_to_read = [
        f"Metadata_Nuclei_TrackObjects_Label",
        "Metadata_number_of_singlecells",
        "Metadata_Well",
        "Metadata_FOV",
        "Metadata_Time",
    ]
    _columns_to_unique_count = [f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}"]
    _actual_column_to_sum = f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}"

elif tracking_type == "LAP":
    _columns_to_read = [
        "Metadata_Nuclei_TrackObjects_Label",
        "Metadata_number_of_singlecells",
        "Metadata_Well",
        "Metadata_FOV",
        "Metadata_Time",
    ]
    _columns_to_unique_count = ["Metadata_Nuclei_TrackObjects_Label"]
    _actual_column_to_sum = "Metadata_Nuclei_TrackObjects_Label"


if tracking_type == "overlap":
    loss = extract_single_time_cell_tracking_entropy(
        df_sc_path=output_file_path,
        columns_to_use=[
            "Metadata_number_of_singlecells",
            "Metadata_dose",
            "Metadata_Time",
            "Metadata_Well",
            "Metadata_FOV",
            f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}",
            "Image_Count_Nuclei",
        ],
        columns_to_groupby=["Metadata_Time", "Metadata_Well", "Metadata_FOV"],
        columns_aggregate_function={
            "Metadata_number_of_singlecells": "mean",
            "Metadata_dose": "first",
            f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei_{max_pixels}": "mean",
            f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei_{max_pixels}": "mean",
            f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei_{max_pixels}": "mean",
            f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei_{max_pixels}": "mean",
            f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}": "max",
            "Image_Count_Nuclei": "mean",
        },
        Max_Cell_Label_col=f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}",
        Lost_Object_Count_col=f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei_{max_pixels}",
        Merged_Object_Count_col=f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei_{max_pixels}",
        New_Object_Count_col=f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei_{max_pixels}",
        Split_Object_Count_col=f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei_{max_pixels}",
        Image_Count_Nuclei_col="Image_Count_Nuclei",
        time_col="Metadata_Time",
        well_col="Metadata_Well",
        fov_col="Metadata_FOV",
        sliding_window_size=2,
    )
elif tracking_type == "LAP":
    loss = extract_single_time_cell_tracking_entropy(
        df_sc_path=output_file_path,
        columns_to_use=[
            "Metadata_number_of_singlecells",
            "Metadata_dose",
            "Metadata_Time",
            "Metadata_Well",
            "Metadata_FOV",
            "Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
            "Metadata_Nuclei_TrackObjects_Label",
            "Image_Count_Nuclei",
        ],
        columns_to_groupby=["Metadata_Time", "Metadata_Well", "Metadata_FOV"],
        columns_aggregate_function={
            "Metadata_number_of_singlecells": "mean",
            "Metadata_dose": "first",
            "Metadata_Image_TrackObjects_LostObjectCount_Nuclei": "mean",
            "Metadata_Image_TrackObjects_MergedObjectCount_Nuclei": "mean",
            "Metadata_Image_TrackObjects_NewObjectCount_Nuclei": "mean",
            "Metadata_Image_TrackObjects_SplitObjectCount_Nuclei": "mean",
            "Metadata_Nuclei_TrackObjects_Label": "max",
            "Image_Count_Nuclei": "mean",
        },
        Max_Cell_Label_col="Metadata_Nuclei_TrackObjects_Label",
        Lost_Object_Count_col="Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
        Merged_Object_Count_col="Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
        New_Object_Count_col="Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
        Split_Object_Count_col="Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
        Image_Count_Nuclei_col="Image_Count_Nuclei",
        time_col="Metadata_Time",
        well_col="Metadata_Well",
        fov_col="Metadata_FOV",
        sliding_window_size=2,
    )


# In[10]:


df_sc = pd.read_parquet("../analysis_output/LAP/Track_Objects_sc.parquet")
df_sc.head()
