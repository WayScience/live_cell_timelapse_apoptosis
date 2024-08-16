"""
This module contains utility functions for setting up the
CellProfiler pipeline for optimization with Optuna.
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the logging format
    handlers=[
        logging.FileHandler("optuna_profiling_utils.log"),  # Log to a file
    ],
)
logger = logging.getLogger(__name__)


# define single functions that can be wrapped in an optuna objective for optimization
def adjust_cppipe_file_LAP(
    trial_number: int,
    cppipe_file_path: pathlib.Path,
    parameters_dict: dict,
) -> pathlib.Path:
    """This function generates a new CellProfiler pipeline file with the parameters from the Optuna trial.
    This function is specific to the LAP cell tracking pipeline.

    Parameters
    ----------
    trial_number : int
        The number of the Optuna trial.
    cppipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    parameters_dict : dict
        A dictionary of the parameters to be optimized.
        This dictionary will have one value per key that are selected from an optimization search space.

    Returns
    -------
    Path to cppipe file : pathlib.Path
    """
    cppipe_file_output_path = pathlib.Path(
        f"{cppipe_file_path.parent}/generated_pipelines/{cppipe_file_path.stem}_trial_{trial_number}.cppipe"
    ).resolve()
    # make the parent directory if it does not exist
    cppipe_file_output_path.parent.mkdir(parents=True, exist_ok=True)

    # load the pipeline as a text file
    with open(cppipe_file_path, "r") as f:
        cppipe_text = f.read()
    # in the cppipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cppipe_text = cppipe_text.replace(
        "Number of standard deviations for search radius:3.0",
        f'Number of standard deviations for search radius:{parameters_dict["std_search_radius"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Search radius limit, in pixel units (Min,Max):2.0,20",
        f'Search radius limit, in pixel units (Min,Max):{parameters_dict["search_radius_min"]},{parameters_dict["search_radius_max"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Gap closing cost:40", f'Gap closing cost:{parameters_dict["gap_closing_cost"]}'
    )
    cppipe_text = cppipe_text.replace(
        "Split alternative cost:40",
        f'Split alternative cost:{parameters_dict["split_alternative_cost"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Merge alternative cost:40",
        f'Merge alternative cost:{parameters_dict["merge_alternative_cost"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Maximum gap displacement, in pixel units:5",
        f'Maximum gap displacement, in pixel units:{parameters_dict["Maximum_gap_displacement"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Maximum split score:50",
        f'Maximum split score:{parameters_dict["Maximum_split_score"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Maximum merge score:50",
        f'Maximum merge score:{parameters_dict["Maximum_merge_score"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Maximum temporal gap, in frames:5",
        f'Maximum temporal gap in frames:{parameters_dict["Maximum_temporal_gap"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Mitosis alternative cost:80",
        f'Mitosis alternative cost:{parameters_dict["Mitosis_alternative_cost"]}',
    )
    cppipe_text = cppipe_text.replace(
        "Maximum mitosis distance, in pixel units:40",
        f'Maximum mitosis distance, in pixel units:{parameters_dict["Maximum_mitosis_distance"]}',
    )

    # write the modified pipeline to a new file
    with open(cppipe_file_output_path, "w") as f:
        f.write(cppipe_text)

    return cppipe_file_output_path


def adjust_cppipe_file_overlap(
    trial_number: int,
    cppipe_file_path: pathlib.Path,
    parameters_dict: dict,
) -> pathlib.Path:
    """This function generates a new CellProfiler pipeline file with the parameters from the Optuna trial.
    This function is specific to the overlap cell tracking pipeline.

    Parameters
    ----------
    trial_number : int
        The number of the Optuna trial.
    cppipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    parameters_dict : dict
        A dictionary of the parameters to be optimized.
        This dictionary will have one value per key that are selected from an optimization search space.

    Returns
    -------
    Path to cppipe file : pathlib.Path
    """
    cppipe_file_output_path = pathlib.Path(
        f"{cppipe_file_path.parent}/generated_pipelines/{cppipe_file_path.stem}_trial_{trial_number}.cppipe"
    ).resolve()
    # make the parent directory if it does not exist
    cppipe_file_output_path.parent.mkdir(parents=True, exist_ok=True)

    # load the pipeline as a text file
    with open(cppipe_file_path, "r") as f:
        cppipe_text = f.read()
    # in the cppipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cppipe_text = cppipe_text.replace(
        "Maximum pixel distance to consider matches:25",
        f'Maximum pixel distance to consider matches:{parameters_dict["Maximum pixel distance to consider matches"]}',
    )

    # write the modified pipeline to a new file
    with open(cppipe_file_output_path, "w") as f:
        f.write(cppipe_text)

    return cppipe_file_output_path


def adjust_cppipe_file_save_tracked_object_images(
    cppipe_file_path: pathlib.Path,
) -> bool:
    """This function updates a cppipe file to enable one module that saves the tracked object images.

    Parameters
    ----------
    cppipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    Returns
    -------
    bool
        True if the pipeline was updated successfully, False otherwise.
    """
    # load the pipeline as a text file
    with open(cppipe_file_path, "r") as f:
        cppipe_text = f.read()
    # in the cppipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cppipe_text = cppipe_text.replace(
        "SaveImages:[module_num:10|svn_version:'Unknown'|variable_revision_number:16|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:False|wants_pause:False]",
        "SaveImages:[module_num:10|svn_version:'Unknown'|variable_revision_number:16|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]",
    )

    # write the modified pipeline to a new file
    with open(cppipe_file_path, "w") as f:
        f.write(cppipe_text)

    return True


def extract_pixel_number_for_overlap_tracking(
    cppipe_file: pathlib.Path,
    search_string: str = "Maximum pixel distance to consider matches:",
) -> Optional[int]:
    """Find the pixel number for overlap tracking in a CellProfiler pipeline file.

    Parameters
    ----------
    cppipe_file : pathlib.Path
        Path of the cppipe file.
    string : _type_, optional
        string to parse the cppipe file fore, by default "Maximum pixel distance to consider matches:"

    Returns
    -------
    int
        The pixel number for overlap tracking. If that is the string is not found, returns None.
    """
    # extract the pixel number from the cppipe file
    with open(cppipe_file, "r") as f:
        lines = f.readlines()
    # find the string in the lines
    for line in lines:
        if search_string in line:
            pixel_number = int(line.split(search_string)[-1])
            return pixel_number
    return None
