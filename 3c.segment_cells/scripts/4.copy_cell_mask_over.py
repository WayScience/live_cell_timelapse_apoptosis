#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import shutil

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")
    parser.add_argument(
        "--final_timepoint_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    parser.add_argument(
        "--terminal_timepoint_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    args = parser.parse_args()
    final_timepoint_dir = pathlib.Path(args.final_timepoint_dir).resolve(strict=True)
    terminal_timepoint_dir = pathlib.Path(args.terminal_timepoint_dir).resolve(
        strict=True
    )


else:
    final_timepoint_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/test_data/timelapse/20231017ChromaLive_6hr_4ch_MaxIP_C-02_F0001"
    ).resolve(strict=True)
    terminal_timepoint_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/test_data/endpoint/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_C-02_F0001"
    ).resolve(strict=True)

well_fov = final_timepoint_dir.name
well_fov = well_fov.split("_")[4] + "_" + well_fov.split("_")[5]


# In[3]:


final_timepoint_cell_mask_path = (
    final_timepoint_dir / f"{well_fov}_T0013_Z0001_cell_mask.tiff"
)
copied_cell_mask_path = (
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_cell_mask.tiff"
)

# copy the cell mask to the terminal timepoint directory
shutil.copy(final_timepoint_cell_mask_path, copied_cell_mask_path)
