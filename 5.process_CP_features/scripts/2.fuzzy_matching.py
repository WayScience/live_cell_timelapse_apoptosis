#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import argparse
import pathlib

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from pycytominer import annotate
from pycytominer.cyto_utils import output

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[2]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# directory where parquet files are located
data_dir = pathlib.Path("../data/1.annotated_data").resolve()

# directory where the annotated parquet files are saved to
profiles_output_dir = pathlib.Path(
    "../data/2.sc_tracks_annotated_data/profiles/"
).resolve()
stats_output_dir = pathlib.Path("../data/2.sc_tracks_annotated_data/stats/").resolve()

profiles_output_dir.mkdir(exist_ok=True, parents=True)
stats_output_dir.mkdir(exist_ok=True, parents=True)

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
    well_fov = "C-02_F0001"


# In[3]:


tracks = pathlib.Path(
    f"../../4.cell_tracking/results/{well_fov}_tracks.parquet"
).resolve(strict=True)
profiles = pathlib.Path(
    f"../data/1.annotated_data/timelapse/{well_fov}_sc.parquet"
).resolve(strict=True)

tracks = pd.read_parquet(tracks)
profiles = pd.read_parquet(
    profiles,
)
# prepend Metadata_ to the tracks columns
tracks.columns = ["Metadata_" + str(col) for col in tracks.columns]
tracks["Metadata_coordinates"] = list(zip(tracks["Metadata_x"], tracks["Metadata_y"]))
profiles["Metadata_coordinates"] = list(
    zip(profiles["Nuclei_AreaShape_Center_X"], profiles["Nuclei_AreaShape_Center_Y"])
)

profiles["Metadata_Time"] = profiles["Metadata_Time"].astype(float)
profiles["Metadata_Time"] = profiles["Metadata_Time"] - 1


# In[4]:


coordinate_column_left = "Metadata_coordinates"
coordinate_column_right = "Metadata_coordinates"
pixel_cutt_off = 5
left_on = ["Metadata_Time"]
right_on = ["Metadata_t"]
merged_df_list = []  # list to store the merged dataframes
total_CP_cells = 0  # total number of cells in the left dataframe
total_annotated_cells = 0  # total number of cells that were annotated
distances = []  # list to store the distances between the coordinates


# In[5]:


tracked_cells_stats = {
    "Metadata_time": [],
    "total_CP_cells": [],
    "total_annotated_cells": [],
}
for time in profiles["Metadata_Time"].unique():
    df_left = profiles.copy().loc[profiles["Metadata_Time"] == time]
    df_right = tracks.copy().loc[tracks["Metadata_t"] == time]

    total_CP_cells += df_left.shape[0]

    # loop through the rows in the subset_annotated_df and find the closest coordinate set in the location metadata
    for index1, row1 in df_left.iterrows():
        tracked_cells_stats["total_CP_cells"].append(1)
        dist = np.inf
        for index2, row2 in df_right.iterrows():
            coord1 = row1[coordinate_column_left]
            coord2 = row2[coordinate_column_right]
            try:
                temp_dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
            except:
                temp_dist = np.inf
            if temp_dist <= dist:
                dist = temp_dist
                coord2_index = index2

            # set cut off of 5,5 pixel in the euclidean distance
            euclidean_cut_off = np.linalg.norm(
                np.array([0, 0]) - np.array([pixel_cutt_off, pixel_cutt_off])
            )

        if dist < euclidean_cut_off:
            temp_merged_df = pd.merge(
                df_left.loc[[index1]],
                df_right.loc[[coord2_index]],
                how="inner",
                left_on=left_on,
                right_on=right_on,
            )
            distances.append(dist)
            total_annotated_cells += temp_merged_df.shape[0]
            tracked_cells_stats["Metadata_time"].append(time)
            tracked_cells_stats["total_annotated_cells"].append(1)
            merged_df_list.append(temp_merged_df)
        else:
            tracked_cells_stats["Metadata_time"].append(time)
            tracked_cells_stats["total_annotated_cells"].append(0)
if len(merged_df_list) == 0:
    merged_df_list.append(pd.DataFrame())
merged_df = pd.concat(merged_df_list)
merged_df["Metadata_distance"] = distances

# replace Metadata string in column names with Metadata (Non Morphology Features)
merged_df.columns = [
    x.replace("Metadata_", "Metadata_") if "Metadata_" in x else x
    for x in merged_df.columns
]

print(f"Annotated cells: {total_annotated_cells} out of {total_CP_cells}")
print(f"Percentage of annotated cells: {total_annotated_cells/total_CP_cells*100}%")
print(merged_df.shape)
merged_df.to_parquet(profiles_output_dir / f"{well_fov}_annotated_tracks.parquet")
merged_df.head()


# In[6]:


# get the number of tracks for each track length
list_of_track_lengths = []
for track in merged_df["Metadata_track_id"].unique():
    track_length = merged_df.loc[merged_df["Metadata_track_id"] == track].shape[0]
    list_of_track_lengths.append(track_length)
list_of_track_lengths_df = pd.DataFrame(list_of_track_lengths, columns=["track_length"])
list_of_track_lengths_df = (
    list_of_track_lengths_df.value_counts().to_frame().reset_index()
)
list_of_track_lengths_df["well_fov"] = well_fov
# save the list of track lengths to a parquet file
list_of_track_lengths_df.to_parquet(
    stats_output_dir / f"{well_fov}_track_lengths.parquet"
)


# In[7]:


tracked_cells_stats_df = pd.DataFrame(tracked_cells_stats)
tracked_cells_stats_df["well_fov"] = well_fov
tracked_cells_stats_df
# get the number of cells for each time point
tracked_cells_stats_df = (
    tracked_cells_stats_df.groupby(["Metadata_time", "well_fov"]).sum().reset_index()
)


# In[8]:


# save the stats to a parquet file
tracked_cells_stats_df.to_parquet(stats_output_dir / f"{well_fov}_stats.parquet")
