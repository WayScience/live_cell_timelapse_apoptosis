"""
This module contains utility functions for the CytoTable conversion
and pycytominer annotation of single cell data extracted from CellProfiler.
"""

import pathlib
import sys

import numpy as np
import optuna
import pandas as pd

sys.path.append("../../utils/")
import logging

import sc_extraction_utils as sc_utils
from cytotable import convert, presets
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from pycytominer import annotate
from pycytominer.cyto_utils import output

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the logging format
    handlers=[
        logging.FileHandler("optuna_profiling_utils.log"),  # Log to a file
    ],
)
logger = logging.getLogger(__name__)


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

        # in place of a preset we will use a custom join statement
        joins = info["preset"]

        logger.info(f"Performing merge single cells and conversion on {sqlite_file}!")

        # merge single cells and output as parquet file
        convert(
            source_path=info["source_path"],
            dest_path=info["dest_path"],
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
        logger.info(f"Merged and converted {pathlib.Path(info['dest_path']).name}!")

        # add single cell count per well as metadata column to parquet file and save back to same path
        sc_utils.add_sc_count_metadata_file(
            data_path=info["dest_path"],
            well_column_name="Image_Metadata_Well",
            file_type="parquet",
        )
        logger.info(
            f"Added single cell count as metadata to {pathlib.Path(info['dest_path']).name}!"
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
        The path to the output file containing the annotated single cell data.
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
