#!/usr/bin/env python
# coding: utf-8

# # Tune the object tracking for cell type

# The goals of this notebook is to run optuna to find the best hyperparameters for object tracking in CellProfiler.
# I will do this by keeping segmentation the same and only changing the tracking parameters.
# I will calculate a loss of the number of cells in a well over time compared to the ground truth of the number of cells in a well over time.

# This optimizes the tracking parameters to get the best tracking results for multiple tracking algorithms.
# The two I will be optimizing are LAP and overlap of objects in the next frame.
# Denoted as LAP and Overlap in the tracking parameters.

# # import libraries

# In[1]:


import pathlib
import pprint
import sys

import numpy as np
import optuna
import pandas as pd
import torch

sys.path.append("../CellProfiler_optuna_utils/")
from cytomining_utils import run_CytoTable, run_pyctominer_annotation
from loss_function_utils import (
    calculate_entropy,
    extract_single_time_cell_tracking_entropy,
    extract_temporal_cell_tracking_entropy,
    harmonic_mean,
    loss_function_from_CP_features,
    loss_function_MSE,
    remove_trial_intermediate_files,
    retrieve_cell_count,
)
from setup_pipeline_utils import (
    adjust_cpipe_file_LAP,
    adjust_cpipe_file_overlap,
    adjust_cpipe_file_save_tracked_object_images,
    extract_pixel_number_for_overlap_tracking,
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


# # set up an argument parser
# parser = argparse.ArgumentParser(description="Run CellProfiler pipelines with Optuna.")
# parser.add_argument(
#     "--tracking_type",
#     "-t",
#     type=str,
#     default="overlap",
#     help="The type of tracking to use. Options are 'overlap' or 'LAP'.",
# )

# parser.add_argument(
#     "--n_trials",
#     "-n",
#     type=int,
#     default=10,
#     help="The number of trials to run.",
# )

# # get the arguments
# args = parser.parse_args()

# # set the tracking type
# tracking_type = args.tracking_type
# n_trials = args.n_trials

tracking_type = "overlap"
n_trials = 1


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


dict_of_inputs_for_cellprofiler = {
    "20231017ChromaLive_6hr_4ch_MaxIP": {
        "path_to_images": pathlib.Path(
            f"{images_dir}/20231017ChromaLive_6hr_4ch_MaxIP/"
        ).resolve(),
        "path_to_output": pathlib.Path("../analysis_output/trial1/").resolve(),
        "path_to_pipeline": pathlib.Path(
            "../pipelines/cell_tracking_optimization_LAP.cppipe"
        ).resolve(),
    },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(dict_of_inputs_for_cellprofiler, indent=4)


# ### CytoTable paths and set up

# In[6]:


# run CytoTable analysis for merged data
# type of file output from CytoTable (currently only parquet)

# dictionary of inputs for CytoTable to pass to the function
dict_of_inputs_for_cytotable = {
    "run_20231004ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            "../analysis_output/trial1/Track_Objects.sqlite"
        ).resolve(),
        "dest_path": pathlib.Path(
            "../analysis_output/trial1/Track_Objects.parquet"
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

# In[7]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# dictionary with each run for the cell type
dict_of_inputs_for_pycytominer = {
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            "../analysis_output/trial1/Track_Objects.parquet"
        ).resolve(),
        "output_file_path": pathlib.Path(
            "../analysis_output/trial1/Track_Objects_sc.parquet"
        ).resolve(),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_6hr_4ch.csv").resolve(
            strict=True
        ),
    }
}


# ### Optuna set up and search space

# In[8]:


# create an optuna parameter search space
# define the search space in a dictionary format
# defined as [min, max] for each parameter
search_space_parameters_LAP = {
    "std_search_radius": [1, 5],  # standard deviation of the search radius
    "search_radius_min": [1, 15],  # minimum search radius in pixels
    "search_radius_max": [16, 100],  # maximum search radius in pixels
    "gap_closing_cost": [1, 100],
    "split_alternative_cost": [1, 100],
    "merge_alternative_cost": [1, 100],
    "Maximum_gap_displacement": [1, 15],
    "Maximum_split_score": [1, 100],
    "Maximum_merge_score": [1, 100],
    "Maximum_temporal_gap": [1, 5],
    "Mitosis_alternative_cost": [1, 50],
    "Maximum_mitosis_distance": [1, 100],  # in pixels
}

# calculate the combination of parameters
combinations = 1
for key in search_space_parameters_LAP.keys():
    combinations *= (
        search_space_parameters_LAP[key][1] - search_space_parameters_LAP[key][0]
    )
print(combinations)


search_space_parameters_overlap = {
    "Maximum pixel distance to consider matches": [1, 201],
}


# set the path to the pipeline file
cpipe_template_file_LAP = pathlib.Path(
    "../pipelines/cell_tracking_optimization_LAP.cppipe"
).resolve(strict=True)
cpipe_template_file_overlap = pathlib.Path(
    "../pipelines/cell_tracking_optimization_overlap.cppipe"
).resolve(strict=True)


# In[9]:


# parameters for the optimization


# this function must be defined in the order of the parameters in the search space
def objective(
    trial: optuna.Trial,
    search_space_parameters_LAP: dict,
    search_space_parameters_overlap: dict,
    dict_of_inputs_for_cellprofiler: dict,
    dict_of_inputs_for_cytotable: dict,
    dict_of_inputs_for_pycytominer: dict,
    cpipe_template_file_LAP: pathlib.Path,
    cpipe_template_file_overlap: pathlib.Path,
    plugins_dir: pathlib.Path,
    tracking_type: str = "LAP",
) -> float:
    """
    This is the objective function that will be optimized by Optuna.
    Here we return the loss function that we want to minimize.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object that will be used to sample the hyperparameters.
    search_space_parameters_LAP : dict
        The dictionary of the search space parameters for the LAP tracking type.
    search_space_parameters_overlap : dict
        The dictionary of the search space parameters for the overlap tracking type.
    dict_of_inputs_for_cellprofiler : dict
        The dictionary of inputs for cellprofiler.
    dict_of_inputs_for_cytotable : dict
        The dictionary of inputs for CytoTable.
    dict_of_inputs_for_pycytominer : dict
        The dictionary of inputs for pycytominer.
    cpipe_template_file_LAP : pathlib.Path
        The path to the LAP pipeline template file.
    cpipe_template_file_overlap : pathlib.Path
        The path to the overlap pipeline template file.
    plugins_dir : pathlib.Path
        The path to the CellProfiler plugins directory.
    tracking_type : str
        The type of tracking to use. Options are 'LAP' or 'overlap'.

    Returns
    -------
    float
        The loss value to minimize.
    """

    # assert that the tracking type is valid
    assert tracking_type in ["LAP", "overlap"], "Invalid tracking type"
    if tracking_type == "LAP":
        dictionary_of_selected_parameters = {
            "std_search_radius": trial.suggest_int(
                "std_search_radius",
                search_space_parameters_LAP["std_search_radius"][0],
                search_space_parameters_LAP["std_search_radius"][1],
            ),
            "search_radius_min": trial.suggest_int(
                "search_radius_min",
                search_space_parameters_LAP["search_radius_min"][0],
                search_space_parameters_LAP["search_radius_min"][1],
            ),
            "search_radius_max": trial.suggest_int(
                "search_radius_max",
                search_space_parameters_LAP["search_radius_max"][0],
                search_space_parameters_LAP["search_radius_max"][1],
            ),
            "gap_closing_cost": trial.suggest_int(
                "gap_closing_cost",
                search_space_parameters_LAP["gap_closing_cost"][0],
                search_space_parameters_LAP["gap_closing_cost"][1],
            ),
            "split_alternative_cost": trial.suggest_int(
                "split_alternative_cost",
                search_space_parameters_LAP["split_alternative_cost"][0],
                search_space_parameters_LAP["split_alternative_cost"][1],
            ),
            "merge_alternative_cost": trial.suggest_int(
                "merge_alternative_cost",
                search_space_parameters_LAP["merge_alternative_cost"][0],
                search_space_parameters_LAP["merge_alternative_cost"][1],
            ),
            "Maximum_gap_displacement": trial.suggest_int(
                "Maximum_gap_displacement",
                search_space_parameters_LAP["Maximum_gap_displacement"][0],
                search_space_parameters_LAP["Maximum_gap_displacement"][1],
            ),
            "Maximum_split_score": trial.suggest_int(
                "Maximum_split_score",
                search_space_parameters_LAP["Maximum_split_score"][0],
                search_space_parameters_LAP["Maximum_split_score"][1],
            ),
            "Maximum_merge_score": trial.suggest_int(
                "Maximum_merge_score",
                search_space_parameters_LAP["Maximum_merge_score"][0],
                search_space_parameters_LAP["Maximum_merge_score"][1],
            ),
            "Maximum_temporal_gap": trial.suggest_int(
                "Maximum_temporal_gap",
                search_space_parameters_LAP["Maximum_temporal_gap"][0],
                search_space_parameters_LAP["Maximum_temporal_gap"][1],
            ),
            "Mitosis_alternative_cost": trial.suggest_int(
                "Mitosis_alternative_cost",
                search_space_parameters_LAP["Mitosis_alternative_cost"][0],
                search_space_parameters_LAP["Mitosis_alternative_cost"][1],
            ),
            "Maximum_mitosis_distance": trial.suggest_int(
                "Maximum_mitosis_distance",
                search_space_parameters_LAP["Maximum_mitosis_distance"][0],
                search_space_parameters_LAP["Maximum_mitosis_distance"][1],
            ),
        }

        # run the pipeline adjustment function
        adjusted_cpipe_file = adjust_cpipe_file_LAP(
            trial_number=trial.number,
            cpipe_file_path=cpipe_template_file_LAP,
            parameters_dict=dictionary_of_selected_parameters,
        )
    elif tracking_type == "overlap":
        dictionary_of_selected_parameters = {
            "Maximum pixel distance to consider matches": trial.suggest_int(
                "Maximum pixel distance to consider matches",
                search_space_parameters_overlap[
                    "Maximum pixel distance to consider matches"
                ][0],
                search_space_parameters_overlap[
                    "Maximum pixel distance to consider matches"
                ][1],
            ),
        }
        # run the pipeline adjustment function
        adjusted_cpipe_file = adjust_cpipe_file_overlap(
            trial_number=trial.number,
            cpipe_file_path=cpipe_template_file_overlap,
            parameters_dict=dictionary_of_selected_parameters,
        )
        max_pixels = dictionary_of_selected_parameters[
            "Maximum pixel distance to consider matches"
        ]

    # update the cellprofiler dictionary with the new pipeline
    dict_of_inputs_for_cellprofiler["20231017ChromaLive_6hr_4ch_MaxIP"][
        "path_to_pipeline"
    ] = adjusted_cpipe_file
    run_name = f"trial_{trial.number}"

    # set the output directory for the trial
    output_dir = pathlib.Path(
        f"../analysis_output/optimized_pipes/{run_name}/"
    ).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # redefine the input and output directories for each step for each trial
    # for cellprofiler, cytotable, and pycytominer
    # cellprofiler
    dict_of_inputs_for_cellprofiler["20231017ChromaLive_6hr_4ch_MaxIP"][
        "path_to_output"
    ] = pathlib.Path(f"{output_dir}/").resolve()

    # cytotable
    dict_of_inputs_for_cytotable["run_20231004ChromaLive_6hr_4ch_MaxIP"][
        "source_path"
    ] = pathlib.Path(f"{output_dir}/Track_Objects.sqlite").resolve()
    dict_of_inputs_for_cytotable["run_20231004ChromaLive_6hr_4ch_MaxIP"][
        "dest_path"
    ] = pathlib.Path(f"{output_dir}/Track_Objects.parquet").resolve()
    # pycytominer
    dict_of_inputs_for_pycytominer["run_20231017ChromaLive_6hr_4ch_MaxIP"][
        "source_path"
    ] = pathlib.Path(f"{output_dir}/Track_Objects.parquet").resolve()
    dict_of_inputs_for_pycytominer["run_20231017ChromaLive_6hr_4ch_MaxIP"][
        "output_file_path"
    ] = pathlib.Path(f"{output_dir}/Track_Objects_sc.parquet").resolve()

    # run cellprofiler pipeline
    cp_parallel.run_cellprofiler_parallel(
        plate_info_dictionary=dict_of_inputs_for_cellprofiler,
        run_name=run_name,
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
    print(f"Trial number: {trial.number}, Loss: {loss}")
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return loss


# ### Run optuna

# In[10]:


# wrap the objective function with multiple arguments in a lambda function
# this needs to be done to pass the additional arguments to the objective function
objective_wrapper = lambda trial: objective(
    trial=trial,
    search_space_parameters_LAP=search_space_parameters_LAP,
    search_space_parameters_overlap=search_space_parameters_overlap,
    dict_of_inputs_for_cellprofiler=dict_of_inputs_for_cellprofiler,
    dict_of_inputs_for_cytotable=dict_of_inputs_for_cytotable,
    dict_of_inputs_for_pycytominer=dict_of_inputs_for_pycytominer,
    cpipe_template_file_LAP=cpipe_template_file_LAP,
    cpipe_template_file_overlap=cpipe_template_file_overlap,
    plugins_dir=plugins_dir,
    tracking_type=tracking_type,
)


# #### The rest of this notebook is not run and is run via a script on the command line.
# I will run with a couple of trials first though.

# In[11]:


# make study directory
study_dir = pathlib.Path("../analysis_output/study_dir").resolve()
study_dir.mkdir(exist_ok=True, parents=True)
# create a study object
if tracking_type == "overlap":
    study = optuna.create_study(
        study_name="cellprofiler_optimization_overlap",
        storage="sqlite:///../analysis_output/study_dir/cellprofiler_optimization_overlap.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    # define study.optimize with the objective function
    study.optimize(objective_wrapper, n_trials=5)

elif tracking_type == "LAP":
    study = optuna.create_study(
        study_name="cellprofiler_optimization",
        storage="sqlite:///../analysis_output/study_dir/cellprofiler_optimization.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    # define study.optimize with the objective function
    study.optimize(objective_wrapper, n_trials=n_trials)


# In[12]:


# get the best parameters
best_params = study.best_params
best_trial = study.best_trial.number
print(f"Best trial: {best_trial}")
print(f"Best loss: {study.best_value}")
print(f"Best parameters: {best_params}")


# In[13]:


if tracking_type == "LAP":
    # move the best trial cpipe file to the main pipeline directory
    current_path = pathlib.Path(
        f"../pipelines/generated_pipelines/cell_tracking_optimization_LAP_trial_{study.best_trial.number}.cppipe"
    ).resolve(strict=True)
    new_path = pathlib.Path(
        "../pipelines/cell_tracking_optimization_LAP_best_trial.cppipe"
    ).resolve()
elif tracking_type == "overlap":
    # move the best trial cpipe file to the main pipeline directory
    current_path = pathlib.Path(
        f"../pipelines/generated_pipelines/cell_tracking_optimization_overlap_trial_{study.best_trial.number}.cppipe"
    ).resolve(strict=True)
    new_path = pathlib.Path(
        "../pipelines/cell_tracking_optimization_overlap_best_trial.cppipe"
    ).resolve()
current_path.rename(new_path)


# In[14]:


# remove trial files
remove_trial_intermediate_files(
    output_dir=pathlib.Path("../analysis_output/optimized_pipes").resolve(),
)

# remove cpipe files
remove_trial_intermediate_files(
    output_dir=pathlib.Path("../pipelines/generated_pipelines").resolve(),
)
