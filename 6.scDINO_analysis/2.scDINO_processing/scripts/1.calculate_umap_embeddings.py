#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import umap

# In[2]:


# set paths
# input path
data_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated.parquet"
).resolve(strict=True)
# output path
output_path = pathlib.Path(
    "../../1.scDINO_run/outputdir/apoptosis_timelapse/CLS_features/CLS_features_annotated_umap.csv"
).resolve()
# shiny output path
shiny_output_path = pathlib.Path(
    "../temporal_shiny_app/CLS_features_annotated_umap.csv"
).resolve()


# In[3]:


# load in data
cls_df = pd.read_parquet(data_path)
cls_df.head()


# In[4]:


# get the metadata
metadata_df = cls_df.columns[cls_df.columns.str.contains("Metadata")]
metadata_df = cls_df[metadata_df]
feature_df = cls_df.drop(metadata_df.columns, axis=1)
print(f"metadata_df shape: {metadata_df.shape}")
print(f"feature_df shape: {feature_df.shape}")


# In[5]:


# define the UMAP model
umap_model = umap.UMAP(
    n_components=2, random_state=0, n_neighbors=30, min_dist=0.1, metric="euclidean"
)

# fit the UMAP model
umap_embedding = umap_model.fit_transform(feature_df)
umap_embedding_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
# add the metadata back
umap_embedding_df = pd.concat([metadata_df, umap_embedding_df], axis=1)
umap_embedding_df.head()


# In[6]:


# save the UMAP embeddings to parquet
umap_embedding_df.to_csv(output_path)

# save the UMAP embeddings to shiny app
umap_embedding_df.to_csv(shiny_output_path)
