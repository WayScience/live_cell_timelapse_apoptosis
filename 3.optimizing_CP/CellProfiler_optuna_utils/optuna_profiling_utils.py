import pathlib
import sys

import numpy as np
import optuna
import pandas as pd
import torch

sys.path.append("../../utils/")
import cp_parallel
from cytotable import convert, presets
from pycytominer import annotate
from pycytominer.cyto_utils import output

sys.path.append("../../utils")
import logging

import sc_extraction_utils as sc_utils
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the logging format
    handlers=[
        logging.FileHandler("optuna_profiling_utils.log"),  # Log to a file
    ],
)
logger = logging.getLogger(__name__)


# define single functions that can be wrapped in an optuna objective for optimization
def adjust_cpipe_file_LAP(
    trial_number: int,
    cpipe_file_path: pathlib.Path,
    parameters_dict: dict,
) -> pathlib.Path:
    """This function generates a new CellProfiler pipeline file with the parameters from the Optuna trial.
    This function is specific to the LAP cell tracking pipeline.

    Parameters
    ----------
    trial_number : int
        The number of the Optuna trial.
    cpipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    parameters_dict : dict
        A dictionary of the parameters to be optimized.
        This dictionary will have one value per key that are selected from an optimization search space.

    Returns
    -------
    Path to cpipe file : pathlib.Path
    """
    cpipe_file_output_path = pathlib.Path(
        f"{cpipe_file_path.parent}/generated_pipelines/{cpipe_file_path.stem}_trial_{trial_number}.cppipe"
    ).resolve()
    # make the parent directory if it does not exist
    cpipe_file_output_path.parent.mkdir(parents=True, exist_ok=True)

    # load the pipeline as a text file
    with open(cpipe_file_path, "r") as f:
        cpipe_text = f.read()
    # in the cpipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cpipe_text = cpipe_text.replace(
        "Number of standard deviations for search radius:3.0",
        f'Number of standard deviations for search radius:{parameters_dict["std_search_radius"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Search radius limit, in pixel units (Min,Max):2.0,20",
        f'Search radius limit, in pixel units (Min,Max):{parameters_dict["search_radius_min"]},{parameters_dict["search_radius_max"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Gap closing cost:40", f'Gap closing cost:{parameters_dict["gap_closing_cost"]}'
    )
    cpipe_text = cpipe_text.replace(
        "Split alternative cost:40",
        f'Split alternative cost:{parameters_dict["split_alternative_cost"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Merge alternative cost:40",
        f'Merge alternative cost:{parameters_dict["merge_alternative_cost"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Maximum gap displacement, in pixel units:5",
        f'Maximum gap displacement, in pixel units:{parameters_dict["Maximum_gap_displacement"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Maximum split score:50",
        f'Maximum split score:{parameters_dict["Maximum_split_score"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Maximum merge score:50",
        f'Maximum merge score:{parameters_dict["Maximum_merge_score"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Maximum temporal gap, in frames:5",
        f'Maximum temporal gap in frames:{parameters_dict["Maximum_temporal_gap"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Mitosis alternative cost:80",
        f'Mitosis alternative cost:{parameters_dict["Mitosis_alternative_cost"]}',
    )
    cpipe_text = cpipe_text.replace(
        "Maximum mitosis distance, in pixel units:40",
        f'Maximum mitosis distance, in pixel units:{parameters_dict["Maximum_mitosis_distance"]}',
    )

    # write the modified pipeline to a new file
    with open(cpipe_file_output_path, "w") as f:
        f.write(cpipe_text)

    return cpipe_file_output_path


def adjust_cpipe_file_overlap(
    trial_number: int,
    cpipe_file_path: pathlib.Path,
    parameters_dict: dict,
) -> pathlib.Path:
    """This function generates a new CellProfiler pipeline file with the parameters from the Optuna trial.
    This function is specific to the overlap cell tracking pipeline.

    Parameters
    ----------
    trial_number : int
        The number of the Optuna trial.
    cpipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    parameters_dict : dict
        A dictionary of the parameters to be optimized.
        This dictionary will have one value per key that are selected from an optimization search space.

    Returns
    -------
    Path to cpipe file : pathlib.Path
    """
    cpipe_file_output_path = pathlib.Path(
        f"{cpipe_file_path.parent}/generated_pipelines/{cpipe_file_path.stem}_trial_{trial_number}.cppipe"
    ).resolve()
    # make the parent directory if it does not exist
    cpipe_file_output_path.parent.mkdir(parents=True, exist_ok=True)

    # load the pipeline as a text file
    with open(cpipe_file_path, "r") as f:
        cpipe_text = f.read()
    # in the cpipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cpipe_text = cpipe_text.replace(
        "Maximum pixel distance to consider matches:25",
        f'Maximum pixel distance to consider matches:{parameters_dict["Maximum pixel distance to consider matches"]}',
    )

    # write the modified pipeline to a new file
    with open(cpipe_file_output_path, "w") as f:
        f.write(cpipe_text)

    return cpipe_file_output_path


def adjust_cpipe_file_save_tracked_object_images(
    cpipe_file_path: pathlib.Path,
) -> None:
    """This function updates a cpipe file to enable one module that saves the tracked object images.

    Parameters
    ----------
    cpipe_file_path : pathlib.Path
        The input CellProfiler pipeline file.
    Returns
    -------
    None
    """
    # load the pipeline as a text file
    with open(cpipe_file_path, "r") as f:
        cpipe_text = f.read()
    # in the cpipe file find the Number of standard deviations for search radius:3.0 line
    # and replace the value with the one from the Optuna trial
    cpipe_text = cpipe_text.replace(
        "SaveImages:[module_num:10|svn_version:'Unknown'|variable_revision_number:16|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:False|wants_pause:False]",
        "SaveImages:[module_num:10|svn_version:'Unknown'|variable_revision_number:16|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]",
    )

    # write the modified pipeline to a new file
    with open(cpipe_file_path, "w") as f:
        f.write(cpipe_text)


def extract_pixel_number_for_overlap_tracking(
    cpipe_file: pathlib.Path,
    string: str = "Maximum pixel distance to consider matches:",
) -> int:
    """Find the pixel number for overlap tracking in a CellProfiler pipeline file.

    Parameters
    ----------
    cpipe_file : pathlib.Path
        Path of the cpipe file.
    string : _type_, optional
        string to parse the cpipe file fore, by default "Maximum pixel distance to consider matches:"

    Returns
    -------
    int
        The pixel number for overlap tracking. If that is the string is not found, returns None.
    """
    string = "Maximum pixel distance to consider matches:"
    # extract the pixel number from the cpipe file
    with open(cpipe_file, "r") as f:
        lines = f.readlines()
    # find the string in the lines
    for line in lines:
        if string in line:
            pixel_number = int(line.split(string)[-1])
            break
    return pixel_number


def run_CytoTable(cytotable_dict: dict, dest_datatype: str = "parquet") -> None:
    """This function runs the CytoTable conversion on the single cell data extracted from CellProfiler.

    Parameters
    ----------
    cytotable_dict : dict
        Dictionary containing the information for the CytoTable conversion.
        The dictionary should have the following structure:
        { run_name: {
            "source_path": source_path, # pathlib.Path
            "dest_path": dest_path, # pathlib.Path
            "preset": preset # str
            }
        }
    dest_datatype : str, optional
        The datatype output to convert the sqlites into, by default "parquet"

    Returns
    -------
    None
    """
    for sqlite_file, info in cytotable_dict.items():
        source_path = info["source_path"]
        dest_path = info["dest_path"]
        # in place of a preset we will use a custom join statement
        joins = info["preset"]

        logger.info(f"Performing merge single cells and conversion on {sqlite_file}!")

        # merge single cells and output as parquet file
        convert(
            source_path=source_path,
            dest_path=dest_path,
            dest_datatype=dest_datatype,
            metadata=["image"],
            compartments=["nuclei"],
            identifying_columns=["ImageNumber"],
            joins=joins,
            parsl_config=Config(
                executors=[HighThroughputExecutor()],
            ),
            chunk_size=1000,
        )
        logger.info(f"Merged and converted {pathlib.Path(dest_path).name}!")

        # add single cell count per well as metadata column to parquet file and save back to same path
        sc_utils.add_sc_count_metadata_file(
            data_path=dest_path,
            well_column_name="Image_Metadata_Well",
            file_type="parquet",
        )
        logger.info(
            f"Added single cell count as metadata to {pathlib.Path(dest_path).name}!"
        )


def run_pyctominer_annotation(
    pycytominer_dict: dict,
) -> pathlib.Path:
    """This function processes the image-based profile data and adds metadata from the platemap file
    to the extracted single cell features.

    Parameters
    ----------
    pycytominer_dict : dict
        This is a dictionary that contains the information for the pycytominer annotation.
        The dictionary should have the following structure:
        { run_name: { "source_path": source_path, "platemap_path": platemap_path } }
        With the following data types:
        run_name: str
        source_path: pathlib.Path
        platemap_path: pathlib.Path

    Returns
    -------
    pathlib.Path
        _description_
    """
    for data_run, info in pycytominer_dict.items():
        # load in converted parquet file as df to use in annotate function
        single_cell_df = pd.read_parquet(info["source_path"])
        platemap_df = pd.read_csv(info["platemap_path"])
        output_file_path = info["output_file_path"]
        logger.info(f"Adding annotations to merged single cells for {data_run}!")

        # add metadata from platemap file to extracted single cell features
        annotated_df = annotate(
            profiles=single_cell_df,
            platemap=platemap_df,
            join_on=["Metadata_well", "Image_Metadata_Well"],
        )

        # move metadata well and single cell count to the front of the df (for easy visualization in python)
        well_column = annotated_df.pop("Metadata_Well")
        singlecell_column = annotated_df.pop("Metadata_number_of_singlecells")
        # insert the column as the second index column in the dataframe
        annotated_df.insert(1, "Metadata_Well", well_column)
        annotated_df.insert(2, "Metadata_number_of_singlecells", singlecell_column)

        # find columns that have path in the name
        file_cols = [col for col in single_cell_df.columns if "FileName" in col]
        path_cols = [col for col in single_cell_df.columns if "PathName" in col]
        # get the cols that contain BoundingBox
        bounding_box_cols = [
            col for col in single_cell_df.columns if "BoundingBox" in col
        ]
        # location cols
        location_cols = [
            "Nuclei_Location_Center_X",
            "Nuclei_Location_Center_Y",
        ]
        track_cols = [col for col in single_cell_df.columns if "TrackObjects" in col]
        # add all lists of columns together
        cols_to_add = (
            file_cols + path_cols + bounding_box_cols + location_cols + track_cols
        )
        logger.info(cols_to_add)

        for col in cols_to_add:
            annotated_df[col] = single_cell_df[col]

        # add "Metadata_" to the beginning of each column if it is in the cols_to_add list
        for col in cols_to_add:
            if col not in annotated_df.columns:
                continue
            if "Metadata_" in col:
                continue
            annotated_df.rename(columns={col: f"Metadata_{col}"}, inplace=True)

        # replace Image_Metadata string in column names with Metadata_
        annotated_df.columns = annotated_df.columns.str.replace(
            "Image_Metadata_", "Metadata_"
        )

        # save annotated df as parquet file
        output(
            df=annotated_df,
            output_filename=output_file_path,
            output_type="parquet",
        )
        logger.info(f"Annotations have been added to {data_run} and saved!")
        # check last annotated df to see if it has been annotated correctly
        logger.info(annotated_df.shape)
        annotated_df.head()
    return output_file_path


def retrieve_cell_count(
    _path_df: pathlib.Path,
    _read_specific_columns: bool = False,
    _columns_to_read: list = None,
    _groupby_columns: list = ["Metadata_Well"],
    _columns_to_unique_count: list = [
        "Metadata_Well",
        "Metadata_Nuclei_TrackObjects_Label",
    ],
    _actual_column_to_sum: str = "Metadata_Nuclei_TrackObjects_Label",
) -> tuple:
    """This function retrieves the actual and target cell counts from a dataframe
    The path supplied should be a parquet file that contains the cell counts from a CellProfiler pipeline
    These actual and target cell counts are used later for the optimization using a loss function

    Parameters
    ----------
    _path_df : pathlib.Path
        This is the path to the parquet file that contains the cell counts
    _read_specific_columns : bool, optional
        This is an option to read the whole parquet file or only certain columns, by default False
    _columns_to_read : list, optional
        This is the specific columns in the parquet to read, by default None
    _groupby_columns : list, optional
        This is the specified columns to group by, by default ['Metadata_Well']
    _columns_to_unique_count : list, optional
        This is the columns to count the unique values of, by default [
        "Metadata_Well",
        "Metadata_Nuclei_TrackObjects_Label",
    ]
    _actual_column_to_sum : str, optional
        This is the actual cell count column to calculate and sum not the target, by default "Metadata_Nuclei_TrackObjects_Label"

    Returns
    -------
    tuple of lists
        Returns a tuple of lists of the actual and target cell counts
    """

    # check if the columns to read are specified
    if _read_specific_columns:
        assert (
            _columns_to_read is not None
        ), "If read_specific_columns is True, columns_to_read must be specified"
        # read in the dataframe
        df = pd.read_parquet(_path_df, columns=_columns_to_read)
    else:
        # read in the dataframe
        df = pd.read_parquet(_path_df)

    logger.info(df.shape)

    ########################################################
    # Get the actual cell counts
    ########################################################

    df_actual_cell_counts = (
        df.groupby(_groupby_columns)[_columns_to_unique_count].nunique().reset_index()
    )
    df_actual_cell_counts = (
        df_actual_cell_counts.groupby("Metadata_Well")[_actual_column_to_sum]
        .sum()
        .reset_index()
    )

    # rename the columns
    df_actual_cell_counts.rename(
        columns={_actual_column_to_sum: "Metadata_number_of_singlecells"},
        inplace=True,
    )

    ########################################################
    # Get the target cell counts
    ########################################################

    # drop all columns except for Metadata_Well and number of single cells
    target_cell_count = df.copy()

    # standard CellProfiler output has the following columns - no need to define dynamically
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


def loss_function_MSE(cell_count: list, target_cell_count: list) -> float:
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
    # mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    loss = mae(cell_count, target_cell_count)
    # loss = mse(cell_count, target_cell_count)
    # convert the loss to a float
    loss = loss.item()
    return loss


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
    df = pd.read_parquet(profile_path)
    array_of_values = df[feature_s_to_use].to_numpy()
    # flatten the array of values
    array_of_values = array_of_values.flatten()

    # calculate the harmonic mean of the array of values

    def harmonic_mean(array: np.array) -> int:
        """This function calculates the harmonic mean of an array of values.

        Returns
        -------
        Float
            The harmonic mean of the array of values.
        """
        array_len = len(array)
        sum_of_reciprocals = 0
        for value in array:
            if value == 0:
                pass
            else:
                sum_of_reciprocals += 1 / value
        harmonic_mean = array_len / sum_of_reciprocals

        return harmonic_mean

    if loss_method == "harmonic_mean":
        loss = harmonic_mean(array_of_values)
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
):
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
            True if the directory is empty, False otherwise
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
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            if is_dir_empty(path):
                path.rmdir()
            else:
                for file in path.iterdir():
                    remove_files(file)
                path.rmdir()

    directories = [str(path) for path in output_dir.rglob("*")]
    count = len(directories)
    while count > 1:
        directories = [str(path) for path in output_dir.rglob("*")]
        count = len(directories)
        for directory in directories:
            print(directory)
            path = pathlib.Path(directory)
            remove_files(path)
