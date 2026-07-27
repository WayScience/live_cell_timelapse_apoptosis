#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from pycytominer import normalize
from pycytominer.cyto_utils import output

# In[2]:


# set  path
normalized_data_path = pathlib.Path(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm.parquet"
).resolve()

endpoint_data_path = pathlib.Path(
    "../../6.process_CP_features/data/4.normalized_data/endpoints/normalized_profile.parquet"
).resolve()

# load data
combined_df = pd.read_parquet(normalized_data_path)
endpoint_df = pd.read_parquet(endpoint_data_path)
combined_df.head()


# In[3]:


endpoint_df.head()


# In[4]:


shared_track_ids = {
    "Metadata_Well": [],
    "Metadata_FOV": [],
    "Metadata_track_id": [],
}
tracks_stats = {
    "Metadata_Well": [],
    "Metadata_FOV": [],
    "number_of_tracks_in_timelapse_data": [],
    "number_of_tracks_in_endpoint_data": [],
    "number_of_common_tracks": [],
    "number_of_lost_tracks": [],
}
for well in combined_df["Metadata_Well"].unique():
    tmp_combined_df = combined_df[combined_df["Metadata_Well"] == well]
    tmp_endpoint_df = endpoint_df[endpoint_df["Metadata_Well"] == well]
    for fov in tmp_combined_df["Metadata_FOV"].unique():
        tmp_combined_df_fov = tmp_combined_df[tmp_combined_df["Metadata_FOV"] == fov]
        tmp_endpoint_df_fov = tmp_endpoint_df[tmp_endpoint_df["Metadata_FOV"] == fov]
        timelapse_track_ids = set(tmp_combined_df_fov["Metadata_track_id"].unique())
        endpoint_track_ids = set(tmp_endpoint_df_fov["Metadata_track_id"].unique())
        common_track_ids = timelapse_track_ids.intersection(endpoint_track_ids)

        shared_track_ids["Metadata_Well"].extend([well] * len(common_track_ids))
        shared_track_ids["Metadata_FOV"].extend([fov] * len(common_track_ids))
        shared_track_ids["Metadata_track_id"].extend(common_track_ids)

        tracks_stats["Metadata_Well"].append(well)
        tracks_stats["Metadata_FOV"].append(fov)
        tracks_stats["number_of_tracks_in_timelapse_data"].append(
            len(timelapse_track_ids)
        )
        tracks_stats["number_of_tracks_in_endpoint_data"].append(
            len(endpoint_track_ids)
        )
        tracks_stats["number_of_common_tracks"].append(len(common_track_ids))
        tracks_stats["number_of_lost_tracks"].append(
            len(timelapse_track_ids) - len(common_track_ids)
        )

tracks_stats_df = pd.DataFrame(tracks_stats)
shared_track_ids_df = pd.DataFrame(shared_track_ids)


# In[5]:


plt.figure(figsize=(10, 6))
# plot a stacked bar chart of number of common tracks and lost tracks for each Well FOV
tracks_stats_df["Well_FOV"] = (
    tracks_stats_df["Metadata_Well"] + "_" + tracks_stats_df["Metadata_FOV"]
)
plt.bar(
    tracks_stats_df["Metadata_Well"],
    tracks_stats_df["number_of_common_tracks"],
    label="Common Tracks",
)
plt.bar(
    tracks_stats_df["Metadata_Well"],
    tracks_stats_df["number_of_lost_tracks"],
    bottom=tracks_stats_df["number_of_common_tracks"],
    label="Lost Tracks",
)
plt.xticks(rotation=90)
plt.xlabel("Well_FOV")
plt.ylabel("Number of Tracks")
plt.title("Number of Common and Lost Tracks for Each Well FOV")
plt.legend()
plt.tight_layout()
pathlib.Path("../figures").mkdir(parents=True, exist_ok=True)
plt.savefig("../figures/track_harmonization/common_and_lost_tracks_by_well_fov.png")
plt.show()


# In[ ]:


# retain only the shared tracks in the combined_df and the endpoint_df
original_combined_df_shape = combined_df.shape
original_endpoint_df_shape = endpoint_df.shape
filtered_combined_df = combined_df.merge(
    shared_track_ids_df,
    on=["Metadata_Well", "Metadata_FOV", "Metadata_track_id"],
    how="inner",
)
filtered_endpoint_df = endpoint_df.merge(
    shared_track_ids_df,
    on=["Metadata_Well", "Metadata_FOV", "Metadata_track_id"],
    how="inner",
)
print(f"Original combined_df shape: {original_combined_df_shape}")
print(f"Original endpoint_df shape: {original_endpoint_df_shape}")
print(f"Filtered combined_df shape: {filtered_combined_df.shape}")
print(f"Filtered endpoint_df shape: {filtered_endpoint_df.shape}")


# In[ ]:


# save the filtered dataframes
filtered_combined_df.to_parquet(
    "../data/CP_scDINO_features/combined_CP_scDINO_norm_filtered_tracks.parquet"
).resolve()
filtered_endpoint_df.to_parquet(
    "../data/CP_scDINO_features/endpoint_CP_norm_filtered_tracks.parquet"
).resolve()
