"""
This module contains utility functions for the optimization
of the CellProfiler pipeline using Optuna.
"""

import logging
import pathlib
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import torch

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the logging format
    handlers=[
        logging.FileHandler("optuna_profiling_utils.log"),  # Log to a file
    ],
)
logger = logging.getLogger(__name__)


def retrieve_cell_count(
    path_df: pathlib.Path,
    read_specific_columns: bool = False,
    columns_to_read: list = None,
    groupby_columns: list = ["Metadata_Well"],
    columns_to_unique_count: list = [
        "Metadata_Well",
        "Metadata_Nuclei_TrackObjects_Label",
    ],
    actual_column_to_sum: str = "Metadata_Nuclei_TrackObjects_Label",
) -> tuple:
    """This function retrieves the actual and target cell counts from a dataframe
    The path supplied should be a parquet file that contains the cell counts from a CellProfiler pipeline
    These actual and target cell counts are used later for the optimization using a loss function

    Parameters
    ----------
    path_df : pathlib.Path
        This is the path to the parquet file that contains the cell counts
    read_specific_columns : bool, optional
        This is an option to read the whole parquet file or only certain columns, by default False
    columns_to_read : list, optional
        This is the specific columns in the parquet to read, by default None
    groupby_columns : list, optional
        This is the specified columns to group by, by default ['Metadata_Well']
    columns_to_unique_count : list, optional
        This is the columns to count the unique values of, by default [
        "Metadata_Well",
        "Metadata_Nuclei_TrackObjects_Label",
    ]
    actual_column_to_sum : str, optional
        This is the actual cell count column to calculate and sum not the target, by default "Metadata_Nuclei_TrackObjects_Label"

    Returns
    -------
    tuple of lists
        Returns a tuple of lists of the actual and target cell counts
    """

    # check if the columns to read are specified
    if read_specific_columns:
        assert (
            columns_to_read is not None
        ), "If read_specific_columns is True, columns_to_read must be specified"
        # read in the dataframe
        df = pd.read_parquet(path_df, columns=columns_to_read)
    else:
        # read in the dataframe
        df = pd.read_parquet(path_df)

    logger.info(df.shape)

    ########################################################
    # Get the actual cell counts
    ########################################################

    df_actual_cell_counts = (
        df.groupby(groupby_columns)[columns_to_unique_count].nunique().reset_index()
    )
    df_actual_cell_counts = (
        df_actual_cell_counts.groupby("Metadata_Well")[actual_column_to_sum]
        .sum()
        .reset_index()
    )

    # rename the columns
    df_actual_cell_counts.rename(
        columns={actual_column_to_sum: "Metadata_number_of_singlecells"},
        inplace=True,
    )

    ########################################################
    # Get the target cell counts
    ########################################################

    # drop all columns except for Metadata_Well and number of single cells
    # standard CellProfiler output has the following columns - no need to define dynamically
    target_cell_count = df.copy()
    target_cell_count = target_cell_count[
        ["Metadata_Well", "Metadata_number_of_singlecells"]
    ]
    target_cell_count.drop_duplicates(inplace=True)

    # sort byt well
    target_cell_count.sort_values(by="Metadata_Well", inplace=True)
    target_cell_count.reset_index(drop=True, inplace=True)

    # get the Metadata_Nuclei_TrackObjects_Label column from the actual cell counts dataframe
    actual_cell_counts = df_actual_cell_counts[
        "Metadata_number_of_singlecells"
    ].to_list()
    target_cell_counts = target_cell_count["Metadata_number_of_singlecells"].to_list()

    return actual_cell_counts, target_cell_counts


def loss_function(cell_count: list, target_cell_count: list) -> float:
    """This is the loss function being used for the optimization of the cell count.
    I am using MSE as a loss function as it is a common loss function for regression problems.

    Parameters
    ----------
    cell_count : list
        A list of the cell counts for each well over time points.
    target_cell_count : list
        A list of the average cell counts for each well over time points.

    Returns
    -------
    loss : float
        The mean squared error between the cell counts and the target cell counts.
    """
    cell_count = torch.tensor(cell_count).float()
    target_cell_count = torch.tensor(target_cell_count).float()

    # calculate the mean squared error using torch
    mae = torch.nn.L1Loss()
    loss = mae(cell_count, target_cell_count)
    # loss = mse(cell_count, target_cell_count)
    # convert the loss to a float
    loss = loss.item()
    return loss


def calculate_harmonic_mean(array: np.array) -> int:
    """This function calculates the harmonic mean of an array of values.

    Returns
    -------
    Float
        The harmonic mean of the array of values.
    """
    # get the length of the array
    array_len = len(array)
    sum_of_reciprocals = 0
    # fine the sum of the reciprocals of the array
    for value in array:
        if value == 0:
            pass
        else:
            sum_of_reciprocals += 1 / value
    # set the harmonic mean to 1000 if the sum of reciprocals is 0
    if sum_of_reciprocals == 0:
        harmonic_mean = 1000
    else:
        harmonic_mean = array_len / sum_of_reciprocals

    return harmonic_mean


def loss_function_from_CP_features(
    profile_path: pathlib.Path,
    feature_s_to_use: list,
    loss_method: str = "harmonic_mean",
) -> float:
    """This is a loss function based directly on the features extracted from CellProfiler.

    Parameters
    ----------
    profile_path : pathlib.Path
        Path to the extracted pycytominer features from CellProfiler.
    feature_s_to_use : list
        feature list to use for the loss function.
    loss_method : str, optional
        Loss type to extract from features, by default "harmonic_mean"

    Returns
    -------
    float
        The calculated loss from the features extracted from CellProfiler.

    Raises
    ------
    ValueError
        If the loss method is not recognized in the conditional statement.
    """

    # read in the dataframe
    df = pd.read_parquet(profile_path, columns=feature_s_to_use)
    array_of_values = df.to_numpy().flatten()

    # calculate the harmonic mean of the array of values

    if loss_method == "harmonic_mean":
        loss = calculate_harmonic_mean(array_of_values)
    elif loss_method == "mean":
        loss = np.mean(array_of_values)
    elif loss_method == "min":
        loss = np.min(array_of_values)
    elif loss_method == "max":
        loss = np.max(array_of_values)
    else:
        raise ValueError("Loss method not recognized")
    return loss


def remove_trial_intermediate_files(
    output_dir: pathlib.Path,
) -> None:
    """This function removes all the intermediate files generated during the optimization process.

    There are also two functions nested within this function that are used to check if a directory is empty
    and to remove all the files in a directory.

    The functions are documented below.

    Parameters
    ----------
    output_dir : pathlib.Path
        Path that contains trial directories with intermediate files.

    Returns
    -------
    None
    """

    def is_dir_empty(path: pathlib.Path) -> bool:
        """this function checks if a directory is empty

        Parameters
        ----------
        path : pathlib.Path
            The path to the directory to check if it is empty

        Returns
        -------
        bool
        """
        if len(list(path.iterdir())) > 0:
            return False
        else:
            return True

    def remove_files(path: pathlib.Path) -> None:
        """This function removes all the files in a directory

        Parameters
        ----------
        path : pathlib.Path
            The path to the directory to remove all the files from

        Returns
        -------
        None
        """
        # if the path is a file, remove it
        if path.is_file():
            path.unlink()
        # if the path is a directory, check if it is empty
        # if it is empty, remove it
        elif path.is_dir():
            if is_dir_empty(path):
                path.rmdir()
            # if it is not empty, remove all the files in the directory
            # and then remove the directory
            else:
                for file in path.iterdir():
                    remove_files(file)
                path.rmdir()

    # get all the directories in the output directory
    directories = [str(path) for path in output_dir.rglob("*")]
    count = len(directories)
    # while there are still directories in the output directory
    while count > 1:
        # update the directories and the count
        directories = [str(path) for path in output_dir.rglob("*")]
        count = len(directories)
        # iterate over each directory and remove all the files
        for directory in directories:
            print(directory)
            path = pathlib.Path(directory)
            remove_files(path)


def calculate_temporal_entropy(
    t1_count: Tuple[int, float],
    t2_count: Tuple[int, float],
) -> float:
    """This function calculates the entropy of two timepoints

    Parameters
    ----------
    t1_count : int | float
        The timepoint one cell count or measurement of choice
    t2_count : int | float
        The timepoint two cell count or measurement of choice

    Returns
    -------
    float
        The calculated entropy of the two timepoints
    """
    # get the total number of cells
    total_cells = t1_count + t2_count
    # calculate the change in cell number
    cell_num_change = abs(t1_count - t2_count)
    # calculate the entropy given the change in cell number and the total number of cells at the time points
    if cell_num_change != 0 and total_cells != 0:
        return float(
            -(cell_num_change / total_cells) * np.log2(cell_num_change / total_cells)
        )
    else:
        return 0.0


def extract_single_time_cell_tracking_entropy(
    df_sc_path: pathlib.Path,
    columns_to_use: list = [
        "Metadata_number_of_singlecells",
        "Metadata_dose",
        "Metadata_Time",
        "Metadata_Well",
        "Metadata_FOV",
        f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
        f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
        f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
        f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
        f"Metadata_Nuclei_TrackObjects_Label",
        "Image_Count_Nuclei",
    ],
    columns_to_groupby: list = ["Metadata_Time", "Metadata_Well", "Metadata_FOV"],
    columns_aggregate_function: dict = {
        "Metadata_number_of_singlecells": "mean",
        "Metadata_dose": "first",
        f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei": "mean",
        f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei": "mean",
        f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei": "mean",
        f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei": "mean",
        f"Metadata_Nuclei_TrackObjects_Label": "max",
        "Image_Count_Nuclei": "mean",
    },
    max_Cell_Label_col: str = f"Metadata_Nuclei_TrackObjects_Label",
    lost_Object_Count_col: str = f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
    merged_Object_Count_col: str = f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
    new_Object_Count_col: str = f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
    split_Object_Count_col: str = f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
    Image_Count_Nuclei_col: str = "Image_Count_Nuclei",
    max_pixels: int = 50,
    time_col: str = "Metadata_Time",
    well_col: str = "Metadata_Well",
    fov_col: str = "Metadata_FOV",
    sliding_window_size: int = 2,
) -> float:
    """This function calculates the entropy of the cell tracking over time for each well and FOV
    The entropy is calculated as the normalized harmonic mean of the entropy other features
    A single mean entropy value is returned

    Parameters
    ----------
    df_sc : pd.DataFrame
        df_sc is a pandas dataframe that contains the single cell data
    columns_to_use : list, optional
        Columns to use to load from the CP pipeline output, by default [
            "Metadata_number_of_singlecells",
            "Metadata_dose",
            "Metadata_Time",
            "Metadata_Well",
            "Metadata_FOV",
            f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
            f"Metadata_Nuclei_TrackObjects_Label",
            "Image_Count_Nuclei",
        ]
    columns_to_groupby : list, optional
        Columns to group by, by default ["Metadata_Time", "Metadata_Well", "Metadata_FOV"]
    columns_aggregate_function : _type_, optional
        A dictionary that identifies which aggregation function to use for each feature, by default {"Metadata_number_of_singlecells": "mean", "Metadata_dose": "first", f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei": "mean", f"Metadata_Nuclei_TrackObjects_Label": "max", "Image_Count_Nuclei": "mean", }
    Max_Cell_Label_col : str, optional
        The name of the exact column that should be used for the Max_Cell_Label_col, by default f"Metadata_Nuclei_TrackObjects_Label"
    Lost_Object_Count_col : str, optional
        The name of the exact column that should be used for the Lost_Object_Count_col, by default f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei"
    Merged_Object_Count_col : str, optional
        The name of the exact column that should be used for the Merged_Object_Count_col, by default f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei"
    New_Object_Count_col : str, optional
        The name of the exact column that should be used for the New_Object_Count_col, by default f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei"
    Split_Object_Count_col : str, optional
        The name of the exact column that should be used for the Split_Object_Count_col, by default f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei"
    Image_Count_Nuclei_col : str, optional
        The name of the exact column that should be used for the Image_Count_Nuclei_col, by default "Image_Count_Nuclei"
    max_pixels : int, optional
        The name of the exact column that should be used for the max_pixels, by default 50
    time_col : str, optional
        The name of the exact column that should be used for the time_col, by default "Metadata_Time"
    well_col : str, optional
        The name of the exact column that should be used for the well_col, by default "Metadata_Well"
    fov_col : str, optional
        The name of the exact column that should be used for the fov_col, by default "Metadata_FOV"
    sliding_window_size : int, optional
        The size of the number of timepoints to include in the sliding window, by default 2

    Returns
    -------
    float
        Return a single float value that represents the mean entropy of the cell tracking over time
    """
    # keep only the columns that are needed
    df_sc = pd.read_parquet(df_sc_path, columns=columns_to_use)

    # aggregate by time, Well, and ImageNumber
    df_sc = (
        df_sc.groupby(columns_to_groupby).agg(columns_aggregate_function).reset_index()
    )
    # sort the values by image number
    # merge the well, time, and image number
    df_sc["Metadata_Well_FOV"] = df_sc[well_col] + "_" + df_sc[fov_col].astype(str)
    df_sc.sort_values(["Metadata_Well_FOV", time_col], inplace=True)
    df_sc

    unique_well_fovs = df_sc["Metadata_Well_FOV"].unique()
    # sliding window entropy calculation for each well and FOV over time
    # get the unique time points
    unique_time_points = df_sc[time_col].unique()

    # aggregate by time, Well, and ImageNumber
    df_sc = (
        df_sc.groupby(["Metadata_Time", "Metadata_Well", "Metadata_FOV"])
        .agg(
            {
                "Metadata_number_of_singlecells": "mean",
                "Metadata_dose": "first",
                max_Cell_Label_col: "max",
                "Image_Count_Nuclei": "mean",
            }
        )
        .reset_index()
    )
    df_sc.rename(
        columns={
            max_Cell_Label_col: "Max_Cell_Label",
            lost_Object_Count_col: "Lost_Object_Count",
            merged_Object_Count_col: "Merged_Object_Count",
            new_Object_Count_col: "New_Object_Count",
            split_Object_Count_col: "Split_Object_Count",
        },
        inplace=True,
    )
    # sort the values by image number
    # merge the well, time, and image number
    df_sc["Metadata_Well_FOV"] = (
        df_sc["Metadata_Well"] + "_" + df_sc["Metadata_FOV"].astype(str)
    )
    df_sc.sort_values(["Metadata_Well_FOV", "Metadata_Time"], inplace=True)
    # iterate over each time point by the sliding window size
    entropy_dict = {
        "time_points": [],
        "well_fov": [],
        "Image_Count_Nuclei": [],
        "Image_Count_Nuclei_entropy": [],
        "Max_Cell_Label": [],
        "difference_in_cell_count": [],
    }
    for timepoint in unique_time_points:
        # get the time points for the sliding window
        # get the data for the time points
        df_sc_timepoint = df_sc.loc[df_sc[time_col] == timepoint]
        for well_fov in unique_well_fovs:
            _df_sc_timepoint = df_sc_timepoint.loc[
                df_sc_timepoint["Metadata_Well_FOV"] == well_fov
            ]

            # input metadata for the entropy calculation
            entropy_dict["time_points"].append(timepoint)
            entropy_dict["well_fov"].append(well_fov)

            ##################################################################
            # Labeled cells vs Image count cells
            ##################################################################
            # print(df_sc_timepoint_1["Image_Count_Nuclei"], df_sc_timepoint_2["Image_Count_Nuclei"])
            entropy_dict["Image_Count_Nuclei"].append(
                _df_sc_timepoint["Image_Count_Nuclei"].values[0]
            )
            entropy_dict["Max_Cell_Label"].append(
                _df_sc_timepoint["Max_Cell_Label"].values[0]
            )
            # calculate the entropy
            entropy_dict["Image_Count_Nuclei_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint["Image_Count_Nuclei"].values[0],
                    _df_sc_timepoint["Max_Cell_Label"].values[0],
                )
            )

            ##################################################################
            # Difference in cell count
            ##################################################################
            # get the cell count for the time points
            entropy_dict["difference_in_cell_count"].append(
                abs(
                    _df_sc_timepoint["Image_Count_Nuclei"].values[0]
                    - _df_sc_timepoint["Max_Cell_Label"].values[0]
                )
            )

    entropy_df = pd.DataFrame.from_dict(entropy_dict)
    # get the mean of the max cell label entropy
    Image_Count_Nuclei_entropy = entropy_df["Image_Count_Nuclei_entropy"].max()
    difference_in_cell_count = entropy_df["difference_in_cell_count"].max()
    # get the harmonic mean of the entropy
    return difference_in_cell_count


def extract_temporal_cell_tracking_entropy(
    df_sc_path: pd.DataFrame,
    columns_to_use: list = [
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
    columns_to_groupby: list = ["Metadata_Time", "Metadata_Well", "Metadata_FOV"],
    columns_aggregate_function: dict = {
        "Metadata_number_of_singlecells": "mean",
        "Metadata_dose": "first",
        "Metadata_Image_TrackObjects_LostObjectCount_Nuclei": "mean",
        "Metadata_Image_TrackObjects_MergedObjectCount_Nuclei": "mean",
        "Metadata_Image_TrackObjects_NewObjectCount_Nuclei": "mean",
        "Metadata_Image_TrackObjects_SplitObjectCount_Nuclei": "mean",
        "Metadata_Nuclei_TrackObjects_Label": "max",
        "Image_Count_Nuclei": "mean",
    },
    max_Cell_Label_col: str = "Metadata_Nuclei_TrackObjects_Label",
    lost_Object_Count_col: str = "Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
    merged_Object_Count_col: str = "Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
    new_Object_Count_col: str = "Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
    split_Object_Count_col: str = "Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
    image_Count_Nuclei_col: str = "Image_Count_Nuclei",
    time_col: str = "Metadata_Time",
    well_col: str = "Metadata_Well",
    fov_col: str = "Metadata_FOV",
    sliding_window_size: int = 2,
) -> float:
    """This function calculates the entropy of the cell tracking over time for each well and FOV
    The entropy is calculated as the normalized harmonic mean of the entropy other features
    A single mean entropy value is returned

    Parameters
    ----------
    df_sc : pd.DataFrame
        df_sc is a pandas dataframe that contains the single cell data
    columns_to_use : list, optional
        Columns to use to load from the CP pipeline output, by default [
            "Metadata_number_of_singlecells",
            "Metadata_dose",
            "Metadata_Time",
            "Metadata_Well",
            "Metadata_FOV",
            f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei",
            f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei",
            f"Metadata_Nuclei_TrackObjects_Label",
            "Image_Count_Nuclei",
        ]
    columns_to_groupby : list, optional
        Columns to group by, by default ["Metadata_Time", "Metadata_Well", "Metadata_FOV"]
    columns_aggregate_function : _type_, optional
        A dictionary that identifies which aggregation function to use for each feature, by default {"Metadata_number_of_singlecells": "mean", "Metadata_dose": "first", f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei": "mean", f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei": "mean", f"Metadata_Nuclei_TrackObjects_Label": "max", "Image_Count_Nuclei": "mean", }
    Max_Cell_Label_col : str, optional
        The name of the exact column that should be used for the Max_Cell_Label_col, by default f"Metadata_Nuclei_TrackObjects_Label"
    Lost_Object_Count_col : str, optional
        The name of the exact column that should be used for the Lost_Object_Count_col, by default f"Metadata_Image_TrackObjects_LostObjectCount_Nuclei"
    Merged_Object_Count_col : str, optional
        The name of the exact column that should be used for the Merged_Object_Count_col, by default f"Metadata_Image_TrackObjects_MergedObjectCount_Nuclei"
    New_Object_Count_col : str, optional
        The name of the exact column that should be used for the New_Object_Count_col, by default f"Metadata_Image_TrackObjects_NewObjectCount_Nuclei"
    Split_Object_Count_col : str, optional
        The name of the exact column that should be used for the Split_Object_Count_col, by default f"Metadata_Image_TrackObjects_SplitObjectCount_Nuclei"
    Image_Count_Nuclei_col : str, optional
        The name of the exact column that should be used for the Image_Count_Nuclei_col, by default "Image_Count_Nuclei"
    max_pixels : int, optional
        The name of the exact column that should be used for the max_pixels, by default 50
    time_col : str, optional
        The name of the exact column that should be used for the time_col, by default "Metadata_Time"
    well_col : str, optional
        The name of the exact column that should be used for the well_col, by default "Metadata_Well"
    fov_col : str, optional
        The name of the exact column that should be used for the fov_col, by default "Metadata_FOV"
    sliding_window_size : int, optional
        The size of the number of timepoints to include in the sliding window, by default 2

    Returns
    -------
    float
        Return a single float value that represents the mean entropy of the cell tracking over time
    """
    df_sc = pd.read_parquet(df_sc_path, columns=columns_to_use)

    # aggregate by time, Well, and ImageNumber
    df_sc = (
        df_sc.groupby(columns_to_groupby).agg(columns_aggregate_function).reset_index()
    )
    df_sc.rename(
        columns={
            max_Cell_Label_col: "Max_Cell_Label",
            lost_Object_Count_col: "Lost_Object_Count",
            merged_Object_Count_col: "Merged_Object_Count",
            new_Object_Count_col: "New_Object_Count",
            split_Object_Count_col: "Split_Object_Count",
        },
        inplace=True,
    )
    # sort the values by image number
    # merge the well, time, and image number
    df_sc["Metadata_Well_FOV"] = df_sc[well_col] + "_" + df_sc[fov_col].astype(str)
    df_sc.sort_values(["Metadata_Well_FOV", time_col], inplace=True)

    unique_well_fovs = df_sc["Metadata_Well_FOV"].unique()
    # sliding window entropy calculation for each well and FOV over time
    # get the unique time points
    unique_time_points = df_sc[time_col].unique()
    # iterate over each time point by the sliding window size
    entropy_dict = {
        "time_points": [],
        "well_fov": [],
        "Image_Count_Nuclei_1": [],
        "Image_Count_Nuclei_2": [],
        "Image_Count_Nuclei_entropy": [],
        "Max_Cell_Label_1": [],
        "Max_Cell_Label_2": [],
        "Max_Cell_Label_entropy": [],
        "Lost_Object_Count_1": [],
        "Lost_Object_Count_2": [],
        "Lost_Object_Count_entropy": [],
        "New_Object_Count_1": [],
        "New_Object_Count_2": [],
        "New_Object_Count_entropy": [],
    }
    for timepoint_window in range(0, len(unique_time_points) - sliding_window_size + 1):
        # get the time points for the sliding window
        time_points = unique_time_points[
            timepoint_window : timepoint_window + sliding_window_size
        ]
        # get the data for the time points
        df_sc_timepoint_1 = df_sc.loc[df_sc[time_col] == time_points[0]]
        df_sc_timepoint_2 = df_sc.loc[df_sc[time_col] == time_points[1]]
        for well_fov in unique_well_fovs:
            _df_sc_timepoint_1 = df_sc_timepoint_1.loc[
                df_sc_timepoint_1["Metadata_Well_FOV"] == well_fov
            ]
            _df_sc_timepoint_2 = df_sc_timepoint_2.loc[
                df_sc_timepoint_2["Metadata_Well_FOV"] == well_fov
            ]

            # input metadata for the entropy calculation
            entropy_dict["time_points"].append(np.mean([int(i) for i in time_points]))
            entropy_dict["well_fov"].append(well_fov)

            ##################################################################
            # Image count nuclei
            ##################################################################
            # print(df_sc_timepoint_1["Image_Count_Nuclei"], df_sc_timepoint_2["Image_Count_Nuclei"])
            entropy_dict["Image_Count_Nuclei_1"].append(
                _df_sc_timepoint_1["Image_Count_Nuclei"].values[0]
            )
            entropy_dict["Image_Count_Nuclei_2"].append(
                _df_sc_timepoint_2["Image_Count_Nuclei"].values[0]
            )
            # calculate the entropy
            entropy_dict["Image_Count_Nuclei_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint_1["Image_Count_Nuclei"].values[0],
                    _df_sc_timepoint_2["Image_Count_Nuclei"].values[0],
                )
            )

            ##################################################################
            # Max cell label
            ##################################################################
            entropy_dict["Max_Cell_Label_1"].append(
                _df_sc_timepoint_1["Max_Cell_Label"].values[0]
            )
            entropy_dict["Max_Cell_Label_2"].append(
                _df_sc_timepoint_2["Max_Cell_Label"].values[0]
            )
            # calculate the entropy
            entropy_dict["Max_Cell_Label_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint_1["Max_Cell_Label"].values[0],
                    _df_sc_timepoint_2["Max_Cell_Label"].values[0],
                )
            )

            ##################################################################
            # Max cell label - image count nuclei
            ##################################################################
            entropy_dict["Max_Cell_Label_1-_Image_Count_Nuclei_1"].append(
                _df_sc_timepoint_1["Max_Cell_Label"].values[0]
                - _df_sc_timepoint_1["Image_Count_Nuclei"].values[0]
            )
            entropy_dict["Max_Cell_Label_2-_Image_Count_Nuclei_2"].append(
                _df_sc_timepoint_2["Max_Cell_Label"].values[0]
                - _df_sc_timepoint_2["Image_Count_Nuclei"].values[0]
            )
            # calculate the entropy
            entropy_dict["Max_Cell_Label-_Image_Count_Nuclei_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint_1["Max_Cell_Label"].values[0]
                    - _df_sc_timepoint_1["Image_Count_Nuclei"].values[0],
                    _df_sc_timepoint_2["Max_Cell_Label"].values[0]
                    - _df_sc_timepoint_2["Image_Count_Nuclei"].values[0],
                )
            )

            ##################################################################
            # Lost object count
            ##################################################################
            entropy_dict["Lost_Object_Count_1"].append(
                _df_sc_timepoint_1["Lost_Object_Count"].values[0]
            )
            entropy_dict["Lost_Object_Count_2"].append(
                _df_sc_timepoint_2["Lost_Object_Count"].values[0]
            )
            # calculate the entropy
            entropy_dict["Lost_Object_Count_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint_1["Lost_Object_Count"].values[0],
                    _df_sc_timepoint_2["Lost_Object_Count"].values[0],
                )
            )

            ##################################################################
            # New object count
            ##################################################################
            entropy_dict["New_Object_Count_1"].append(
                _df_sc_timepoint_1["New_Object_Count"].values[0]
            )
            entropy_dict["New_Object_Count_2"].append(
                _df_sc_timepoint_2["New_Object_Count"].values[0]
            )
            # calculate the entropy
            entropy_dict["New_Object_Count_entropy"].append(
                calculate_temporal_entropy(
                    _df_sc_timepoint_1["New_Object_Count"].values[0],
                    _df_sc_timepoint_2["New_Object_Count"].values[0],
                )
            )

    entropy_df = pd.DataFrame.from_dict(entropy_dict)
    # get the harmonic mean of the entropy
    entropy_df["harmonic_mean_entropy"] = entropy_df[
        [
            # "Image_Count_Nuclei_entropy",
            "Max_Cell_Label_entropy",
            "Lost_Object_Count_entropy",
            "New_Object_Count_entropy",
        ]
    ].apply(lambda x: calculate_harmonic_mean(x), axis=1)
    # calculate the adjusted entropy for each time point pair
    # the adjusted entropy is the
    # absolute difference of the entropy values of each time point pair
    # for each feature
    # compared to that of the Max Cell Label feature
    # this grounds the entropy values in realtion to the entropy of segmentation
    Max_Cell_Label = entropy_df["Max_Cell_Label"].max()
    # normalized entropy values
    entropy_df["noramlized_Max_Cell_Label_entropy__to__Image_Count_Nuclei_entropy"] = (
        entropy_df["Max_Cell_Label_entropy"] / entropy_df["Image_Count_Nuclei_entropy"]
    )
    entropy_df["noramlized_harmonic_mean_entropy__to__Image_Count_Nuclei_entropy"] = (
        entropy_df["harmonic_mean_entropy"] / entropy_df["Image_Count_Nuclei_entropy"]
    )
    entropy = entropy_df["Max_Cell_Label-_Image_Count_Nuclei_entropy"].max()

    return Max_Cell_Label
