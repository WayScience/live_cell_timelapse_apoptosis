import pathlib
import sys

import optuna
import pandas as pd
import torch

sys.path.append("../../utils/")
import cp_parallel
from cytotable import convert, presets
from pycytominer import annotate
from pycytominer.cyto_utils import output

sys.path.append("../../utils")
import sc_extraction_utils as sc_utils
from parsl.config import Config
from parsl.executors import HighThroughputExecutor


# define single functions that can be wrapped in an optuna objective for optimization
def adjust_cpipe_file(
    trial_number: int, cpipe_file_path: pathlib.Path, parameters_dict: dict
) -> pathlib.Path:
    """This function generates a new CellProfiler pipeline file with the parameters from the Optuna trial.

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
    pathlib.Path
    """
    cpipe_file_output_path = pathlib.Path(
        f'{cpipe_file_path.parent}/generated_pipelines/{cpipe_file_path.name.strip(".cppipe")}_trial_{trial_number}.cppipe'
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


def run_CytoTable(cytotable_dict: dict, dest_datatype: str = "parquet"):
    for sqlite_file, info in cytotable_dict.items():
        source_path = info["source_path"]
        dest_path = info["dest_path"]
        preset = info["preset"]
        presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"] = info[
            "preset"
        ]
        print(f"Performing merge single cells and conversion on {sqlite_file}!")

        # merge single cells and output as parquet file
        convert(
            source_path=source_path,
            dest_path=dest_path,
            dest_datatype=dest_datatype,
            preset=preset,
            parsl_config=Config(
                executors=[HighThroughputExecutor()],
            ),
            chunk_size=1000,
        )
        print(f"Merged and converted {pathlib.Path(dest_path).name}!")

        # add single cell count per well as metadata column to parquet file and save back to same path
        sc_utils.add_sc_count_metadata_file(
            data_path=dest_path,
            well_column_name="Image_Metadata_Well",
            file_type="parquet",
        )
        print(f"Added single cell count as metadata to {pathlib.Path(dest_path).name}!")


def run_pyctominer_annotation(pycytominer_dict: dict, output_dir: str) -> pathlib.Path:
    for data_run, info in pycytominer_dict.items():
        # load in converted parquet file as df to use in annotate function
        single_cell_df = pd.read_parquet(info["source_path"])
        platemap_df = pd.read_csv(info["platemap_path"])
        output_file = str(pathlib.Path(f"{output_dir}/{data_run}_sc.parquet"))
        print(f"Adding annotations to merged single cells for {data_run}!")

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
        # add all lists of columns together
        cols_to_add = file_cols + path_cols + bounding_box_cols + location_cols
        print(cols_to_add)

        for col in cols_to_add:
            annotated_df[col] = single_cell_df[col]

        # add "Metadata_" to the beginning of each column if it is in the cols_to_add list
        for col in cols_to_add:
            if col not in annotated_df.columns:
                continue
            if "Metadata_" in col:
                continue
            annotated_df.rename(columns={col: f"Metadata_{col}"}, inplace=True)

        # save annotated df as parquet file
        output(
            df=annotated_df,
            output_filename=output_file,
            output_type="parquet",
        )
        print(f"Annotations have been added to {data_run} and saved!")
        # check last annotated df to see if it has been annotated correctly
        print(annotated_df.shape)
        annotated_df.head()
    return output_file


def retrieve_cell_count():
    pass


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
    mse = torch.nn.MSELoss()
    loss = mse(cell_count, target_cell_count)
    # convert the loss to a float
    loss = loss.item()
    return loss
