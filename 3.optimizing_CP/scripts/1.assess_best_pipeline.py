#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# In[ ]:


# set up an argument parser
parser = argparse.ArgumentParser(description="Run CellProfiler pipelines with Optuna.")
parser.add_argument(
    "--tracking_type",
    "-t",
    type=str,
    default="overlap",
    help="The type of tracking to use. Options are 'overlap' or 'LAP'.",
)

# get the arguments
args = parser.parse_args()

# set the tracking type
tracking_type = args.tracking_type


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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
        f"Metadata_Nuclei_TrackObjects_Label_{max_pixels}",
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

# retrieve the cell counts
actual_cell_counts, target_cell_counts = retrieve_cell_count(
    _path_df=output_file_path,
    _read_specific_columns=True,
    _columns_to_read=_columns_to_read,
    _groupby_columns=[
        "Metadata_Well",
        "Metadata_Time",
    ],
    _columns_to_unique_count=_columns_to_unique_count,
    _actual_column_to_sum=_actual_column_to_sum,
)
if tracking_type == "overlap":

    # calculate the loss function
    loss = loss_function_from_CP_features(
        profile_path=output_file_path,
        loss_method="harmonic_mean",
        feature_s_to_use=[
            f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei_{max_pixels}",
            f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei_{max_pixels}",
        ],
    )
elif tracking_type == "LAP":
    # calculate the loss function
    loss = loss_function_from_CP_features(
        profile_path=output_file_path,
        loss_method="harmonic_mean",
        feature_s_to_use=[
            "Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
            "Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
        ],
    )


# In[ ]:


# # remove trial files
# remove_trial_intermediate_files(
#     output_dir = pathlib.Path("../analysis_output/").resolve(),
# )
