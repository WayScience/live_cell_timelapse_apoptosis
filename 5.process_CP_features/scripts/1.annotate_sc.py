#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import pathlib

import lancedb
import pandas as pd
from pycytominer import annotate
from pycytominer.cyto_utils import output

# ## Set paths and variables

# In[2]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path("../../data/").resolve()

# directory where parquet files are located
data_dir = pathlib.Path("../data/converted_data")

# directory where the annotated parquet files are saved to
output_dir = pathlib.Path("../data/annotated_data")
output_dir.mkdir(exist_ok=True)


# In[3]:


# dictionary with each run for the cell type
dict_of_inputs = {
    "run_20231017ChromaLive_6hr_4ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/20231017ChromaLive_6hr_4ch_MaxIP.parquet"
        ).resolve(strict=True),
        "platemap_path": pathlib.Path(f"{platemap_path}/platemap_6hr_4ch.csv").resolve(
            strict=True
        ),
    },
    "20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP": {
        "source_path": pathlib.Path(
            f"{data_dir}/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP.parquet"
        ).resolve(strict=True),
        "platemap_path": pathlib.Path(
            f"{platemap_path}/platemap_AnnexinV_2ch.csv"
        ).resolve(strict=True),
    },
}


# ## Annotate merged single cells

# In[4]:


for data_run, info in dict_of_inputs.items():
    # load in converted parquet file as df to use in annotate function
    single_cell_df = pd.read_parquet(info["source_path"])
    print(single_cell_df.shape)
    single_cell_df.rename(
        columns={
            "Image_Metadata_FOV": "Metadata_FOV",
            "Image_Metadata_Time": "Metadata_Time",
        },
        inplace=True,
    )
    platemap_df = pd.read_csv(info["platemap_path"])
    output_file = str(pathlib.Path(f"{output_dir}/{data_run}_sc.parquet"))
    print(f"Adding annotations to merged single cells for {data_run}!")

    # add metadata from platemap file to extracted single cell features
    annotated_df = annotate(
        profiles=single_cell_df,
        platemap=platemap_df,
        join_on=["Metadata_well", "Image_Metadata_Well"],
    )
    print(annotated_df.shape)

    # move metadata well and single cell count to the front of the df (for easy visualization in python)
    well_column = annotated_df.pop("Metadata_Well")
    singlecell_column = annotated_df.pop("Metadata_number_of_singlecells")
    # insert the column as the second index column in the dataframe
    annotated_df.insert(1, "Metadata_Well", well_column)
    annotated_df.insert(2, "Metadata_number_of_singlecells", singlecell_column)

    # rename metadata columns to match the expected column names
    columns_to_rename = {
        "Nuclei_Location_Center_Y": "Metadata_Nuclei_Location_Center_Y",
        "Nuclei_Location_Center_X": "Metadata_Nuclei_Location_Center_X",
    }
    # Image_FileName cols
    for col in annotated_df.columns:
        if "Image_FileName" in col:
            columns_to_rename[col] = f"Metadata_{col}"
        elif "Image_PathName" in col:
            columns_to_rename[col] = f"Metadata_{col}"
        elif "TrackObjects" in col:
            columns_to_rename[col] = f"Metadata_{col}"
    # rename metadata columns
    annotated_df.rename(columns=columns_to_rename, inplace=True)

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


# ### Merge the terminal and single cell data

# ## Add the object tacking from SAM2

# In[5]:


# set and connect to the db
# create the database object
uri = pathlib.Path("../../data/objects_db").resolve()
db = lancedb.connect(uri)


# In[6]:


# get the db schema and tables
db.table_names()
# load table
table = db["1.masked_images"]
location_metadata_df = table.to_pandas()
print(location_metadata_df.shape)
location_metadata_df.head()


# In[7]:


# change frame to Metadata_Time
location_metadata_df.rename(columns={"frame": "Metadata_Time"}, inplace=True)
# add 1 to Metadata_Time to match the timepoints in the single cell data
location_metadata_df["Metadata_Time"] = location_metadata_df["Metadata_Time"] + 1
# change formatting to leading 4 zeros
location_metadata_df["Metadata_Time"] = location_metadata_df["Metadata_Time"].apply(
    lambda x: f"{x:04}"
)
print(location_metadata_df.shape)
location_metadata_df.head()


# ### Loop through the saved annotated dfs and add the object tracking

# In[8]:


# for data_run in dict_of_inputs.keys():
#     # load in annotated parquet file as df to use in annotate function
#     annotated_df = pd.read_parquet(
#         pathlib.Path(f"{output_dir}/{data_run}_sc.parquet").resolve(strict=True)
#     )
#     print(f"Oringinal shape of {data_run} is {annotated_df.shape}")
#     print(f"Adding location metadata to single cells for {data_run}!")

#     annotated_df["Metadata_image_set_name"] = (
#         annotated_df["Metadata_Well"].astype(str)
#         + "_"
#         + "F"
#         + annotated_df["Metadata_FOV"].astype(str)
#     )
#     image_set_names = annotated_df.pop("Metadata_image_set_name")
#     # move to front
#     annotated_df.insert(0, "Metadata_image_set_name", image_set_names)
#     time = annotated_df.pop("Metadata_Time")
#     annotated_df.insert(1, "Metadata_Time", time)
#     x_coord = annotated_df.pop("Metadata_Nuclei_Location_Center_X")
#     Y_coord = annotated_df.pop("Metadata_Nuclei_Location_Center_Y")
#     annotated_df.insert(2, "Metadata_Nuclei_Location_Center_X", x_coord)
#     annotated_df.insert(3, "Metadata_Nuclei_Location_Center_Y", Y_coord)
#     annotated_df.head()
#     # drop NaN values in the centroid columns from annotated_df
#     annotated_df = annotated_df.dropna(
#         subset=["Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y"]
#     )
#     # match the x and y coordinates to the image set name in the location metadata df
#     annotated_df["Metadata_Nuclei_Location_Center_X"] = annotated_df[
#         "Metadata_Nuclei_Location_Center_X"
#     ].astype(int)
#     annotated_df["Metadata_Nuclei_Location_Center_Y"] = annotated_df[
#         "Metadata_Nuclei_Location_Center_Y"
#     ].astype(int)
#     location_metadata_df["x"] = location_metadata_df["x"].astype(int)
#     location_metadata_df["y"] = location_metadata_df["y"].astype(int)

#     merged_df = annotated_df.merge(
#         location_metadata_df,
#         how="left",
#         left_on=[
#             "Metadata_Nuclei_Location_Center_X",
#             "Metadata_Nuclei_Location_Center_Y",
#             "Metadata_Time",
#             "Metadata_image_set_name",
#         ],
#         right_on=["x", "y", "Metadata_Time", "image_set_name"],
#     )
#     # sort by image_set_name and Metadata_Time
#     merged_df = merged_df.sort_values(by=["Metadata_image_set_name", "Metadata_Time"])
#     # drop right columns
#     merged_df = merged_df.drop(
#         columns=[
#             "image_set_name",
#             "object_id",
#             "x",
#             "y",
#             "mask_path",
#             "mask_file_name",
#             "mask_file_path",
#         ]
#     )
#     print(f"The final merged shape of {data_run} is {merged_df.shape}")
#     # save annotated df as parquet file
#     output(
#         df=merged_df,
#         output_filename=output_file,
#         output_type="parquet",
#     )


# In[9]:


# print(annotated_df.shape)
# annotated_df["Metadata_image_set_name"] = (
#     annotated_df["Metadata_Well"].astype(str)
#     + "_"
#     + "F"
#     + annotated_df["Metadata_FOV"].astype(str)
# )
# image_set_names = annotated_df.pop("Metadata_image_set_name")
# # move to front
# annotated_df.insert(0, "Metadata_image_set_name", image_set_names)
# time = annotated_df.pop("Metadata_Time")
# annotated_df.insert(1, "Metadata_Time", time)
# x_coord = annotated_df.pop("Metadata_Nuclei_Location_Center_X")
# Y_coord = annotated_df.pop("Metadata_Nuclei_Location_Center_Y")
# annotated_df.insert(2, "Metadata_Nuclei_Location_Center_X", x_coord)
# annotated_df.insert(3, "Metadata_Nuclei_Location_Center_Y", Y_coord)
# annotated_df.head()


# In[10]:


# # drop NaN values in the centroid columns from annotated_df
# print(annotated_df.shape)
# annotated_df = annotated_df.dropna(
#     subset=["Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y"]
# )
# print(annotated_df.shape)
# print(location_metadata_df.shape)
# # match the x and y coordinates to the image set name in the location metadata df
# annotated_df["Metadata_Nuclei_Location_Center_X"] = annotated_df[
#     "Metadata_Nuclei_Location_Center_X"
# ].astype(int)
# annotated_df["Metadata_Nuclei_Location_Center_Y"] = annotated_df[
#     "Metadata_Nuclei_Location_Center_Y"
# ].astype(int)
# location_metadata_df["x"] = location_metadata_df["x"].astype(int)
# location_metadata_df["y"] = location_metadata_df["y"].astype(int)

# merged_df = annotated_df.merge(
#     location_metadata_df,
#     how="left",
#     left_on=[
#         "Metadata_Nuclei_Location_Center_X",
#         "Metadata_Nuclei_Location_Center_Y",
#         "Metadata_Time",
#         "Metadata_image_set_name",
#     ],
#     right_on=["x", "y", "Metadata_Time", "image_set_name"],
# )
# print(merged_df.shape)
# # sort by image_set_name and Metadata_Time
# merged_df = merged_df.sort_values(by=["Metadata_image_set_name", "Metadata_Time"])
# # drop right columns
# merged_df = merged_df.drop(
#     columns=[
#         "image_set_name",
#         "object_id",
#         "x",
#         "y",
#         "mask_path",
#         "mask_file_name",
#         "mask_file_path",
#     ]
# )
# merged_df.head()


# In[11]:


# # save annotated df as parquet file
# output(
#     df=merged_df,
#     output_filename=output_file,
#     output_type="parquet",
# )
