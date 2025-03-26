#!/usr/bin/env python
# coding: utf-8

# # Merge single cells from CellProfiler outputs using CytoTable

# In[1]:


import argparse
import pathlib
import sys

import pandas as pd
from cytotable import convert, presets

sys.path.append("../../utils")
import sc_extraction_utils as sc_utils
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables
#
# All paths must be string but we use pathlib to show which variables are paths

# In[2]:


# type of file output from CytoTable (currently only parquet)
dest_datatype = "parquet"

# s1lite directory
source_dir = pathlib.Path("../../4.cellprofiler_analysis/analysis_output/").resolve(
    strict=True
)
# directory where parquet files are saved to
output_dir = pathlib.Path("../data/0.converted_data")
output_dir.mkdir(exist_ok=True, parents=True)

if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Single cell extraction")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
else:
    print("Running in a notebook")
    well_fov = "E-05_F0003"


# ## set config joins for each preset

# In[3]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"


# In[4]:


dict_of_inputs = {
    "20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{source_dir}/timelapse/{well_fov}/timelapse_4ch_analysis.sqlite"
        ).resolve(strict=True),
        "dest_path": pathlib.Path(
            f"{output_dir}/timelapse/{well_fov}.parquet"
        ).resolve(),
        "preset": """WITH Per_Image_Filtered AS (
                SELECT
                    Metadata_ImageNumber,
                    Image_Metadata_Well,
                    Image_Metadata_FOV,
                    Image_Metadata_Time,
                    Image_PathName_CL_488_1
                    Image_PathName_CL_488_2,
                    Image_PathName_CL_561,
                    Image_FileName_CL_488_1,
                    Image_FileName_CL_488_2,
                    Image_FileName_CL_561,
                    Image_FileName_DNA,

                FROM
                    read_parquet('per_image.parquet')
                )
            SELECT
                *
            FROM
                Per_Image_Filtered AS per_image
            LEFT JOIN read_parquet('per_cytoplasm.parquet') AS per_cytoplasm ON
                per_cytoplasm.Metadata_ImageNumber = per_image.Metadata_ImageNumber
            LEFT JOIN read_parquet('per_cells.parquet') AS per_cells ON
                per_cells.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_cells.Metadata_Cells_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Cells
            LEFT JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_nuclei.Metadata_Nuclei_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
                """,
    },
    "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{source_dir}/endpoint/{well_fov}/timelapse_2ch_analysis.sqlite"
        ).resolve(),
        "dest_path": pathlib.Path(
            f"{output_dir}/endpoint/{well_fov}.parquet"
        ).resolve(),
        "preset": """WITH Per_Image_Filtered AS (
                SELECT
                    Metadata_ImageNumber,
                    Image_Metadata_Well,
                    Image_Metadata_FOV,
                    Image_Metadata_Time,
                    Image_PathName_AnnexinV,
                    Image_PathName_DNA,
                    Image_FileName_AnnexinV,
                    Image_FileName_DNA


                FROM
                    read_parquet('per_image.parquet')
                )
            SELECT
                *
            FROM
                Per_Image_Filtered AS per_image
            LEFT JOIN read_parquet('per_cytoplasm.parquet') AS per_cytoplasm ON
                per_cytoplasm.Metadata_ImageNumber = per_image.Metadata_ImageNumber
            LEFT JOIN read_parquet('per_cells.parquet') AS per_cells ON
                per_cells.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_cells.Metadata_Cells_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Cells
            LEFT JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_nuclei.Metadata_Nuclei_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
                """,
    },
    "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_whole_image": {
        "source_path": pathlib.Path(
            f"{source_dir}/endpoint/{well_fov}/timelapse_2ch_analysis.sqlite"
        ).resolve(),
        "dest_path": pathlib.Path(
            f"{output_dir}/endpoint/{well_fov}_whole_image.parquet"
        ).resolve(),
        "preset": """WITH Per_Image_Filtered AS (
                SELECT
                    Metadata_ImageNumber,
                    Image_Metadata_Well,
                    Image_Metadata_FOV,
                    Image_Metadata_Time,
                    Image_PathName_AnnexinV,
                    Image_PathName_DNA,
                    Image_FileName_AnnexinV,
                    Image_FileName_DNA


                FROM
                    read_parquet('per_image.parquet')
                )
            SELECT
                *
            FROM
                Per_Image_Filtered AS per_image
            LEFT JOIN read_parquet('per_cytoplasm.parquet') AS per_cytoplasm ON
                per_cytoplasm.Metadata_ImageNumber = per_image.Metadata_ImageNumber
            LEFT JOIN read_parquet('per_cells.parquet') AS per_cells ON
                per_cells.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_cells.Metadata_Cells_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Cells
            LEFT JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_nuclei.Metadata_Nuclei_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
                """,
    },
}


# ## Convert SQLite file and merge single cells into parquet file
#
# This was not run to completion as we use the nbconverted python file for full run.

# In[5]:


# run through each run with each set of paths based on dictionary
for sqlite_file, info in dict_of_inputs.items():
    source_path = info["source_path"]
    dest_path = info["dest_path"]
    presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"] = info["preset"]
    print(f"Performing merge single cells and conversion on {sqlite_file}!")
    print(f"Source path: {source_path}")
    print(f"Destination path: {dest_path}")
    # merge single cells and output as parquet file
    try:
        convert(
            source_path=source_path,
            dest_path=dest_path,
            dest_datatype=dest_datatype,
            preset=preset,
            parsl_config=Config(
                executors=[HighThroughputExecutor()],
            ),
            chunk_size=10000,
        )

        print(f"Merged and converted {pathlib.Path(dest_path).name}!")
        df = pd.read_parquet(dest_path)
        print(f"Shape of {pathlib.Path(dest_path).name}: {df.shape}")
        # add single cell count per well as metadata column to parquet file and save back to same path
        sc_utils.add_sc_count_metadata_file(
            data_path=dest_path,
            well_column_name="Metadata_ImageNumber",
            file_type="parquet",
        )

        # read the parquet file to check if metadata was added
        df1 = pd.read_parquet(dest_path)
        print(f"Shape of {pathlib.Path(dest_path).name}: {df.shape}")
        print(f"Added single cell count as metadata to {pathlib.Path(dest_path).name}!")
    except Exception as e:
        print(f"Error in merging and converting {sqlite_file}!")
        print("Probably due to no objects recorded in the sqlite file")
        print(e)
        continue


# In[ ]:
