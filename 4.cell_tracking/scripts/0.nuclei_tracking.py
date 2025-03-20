#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from PIL import Image
from rich.pretty import pprint
from ultrack import to_tracks_layer, track, tracks_to_zarr
from ultrack.config import MainConfig
from ultrack.tracks import close_tracks_gaps
from ultrack.utils import estimate_parameters_from_labels, labels_to_contours

# check if in a jupyter notebook

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(f"Running in notebook: {in_notebook}")

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# check gpu
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if not gpu_devices:
    print("No GPU found")
else:
    print("GPU found")


# tensorflow clear gpu memory
def clear_gpu_memory():
    from numba import cuda

    cuda.select_device(0)
    cuda.close()


clear_gpu_memory()
import napari
from napari.utils.notebook_display import nbscreenshot


# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)

else:
    print("Running in a notebook")
    input_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/test_data/timelapse/20231017ChromaLive_6hr_4ch_MaxIP_C-02_F0001"
    ).resolve(strict=True)

temporary_output_dir = pathlib.Path("../tmp_output").resolve()
figures_output_dir = pathlib.Path("../figures").resolve()
results_output_dir = pathlib.Path("../results").resolve()
temporary_output_dir.mkdir(exist_ok=True)
figures_output_dir.mkdir(exist_ok=True)
results_output_dir.mkdir(exist_ok=True)


# In[3]:


print(f"Input directory: {input_dir}")


# In[4]:


file_extensions = {".tif", ".tiff"}
# get all the tiff files
tiff_files = list(input_dir.glob("*"))
tiff_files = [f for f in tiff_files if f.suffix in file_extensions]
tiff_files = sorted(tiff_files)

mask_files = [f for f in tiff_files if "nuclei" in f.name]
nuclei_files = [f for f in tiff_files if "C01" in f.name]

print(f"Found {len(mask_files)} tiff files in the input directory")
print(f"Found {len(nuclei_files)} nuclei files in the input directory")


# In[5]:


# read in the masks and create labels
masks = []
for tiff_file in mask_files:
    img = tifffile.imread(tiff_file)
    masks.append(img)
masks = np.array(masks)

nuclei = []
for tiff_file in nuclei_files:
    img = tifffile.imread(tiff_file)
    nuclei.append(img)
nuclei = np.array(nuclei)


# In[6]:


image_dims = tifffile.imread(tiff_files[0]).shape


# In[7]:


detections = np.zeros((len(masks), image_dims[0], image_dims[1]), dtype=np.uint16)
edges = np.zeros((len(masks), image_dims[0], image_dims[1]), dtype=np.uint16)
for frame_index, frame in enumerate(masks):
    detections[frame_index, :, :], edges[frame_index, :, :] = labels_to_contours(
        frame
    )  # gets the contours of the masks and the edges
print(detections.shape, edges.shape)

clear_gpu_memory()


# In[8]:


params_df = estimate_parameters_from_labels(masks, is_timelapse=True)
if in_notebook:
    params_df["area"].plot(kind="hist", bins=100, title="Area histogram")


# ## Optimize the tracking using ultrack

# In[ ]:





# In[9]:


config = MainConfig()
config.linking_config.max_distance = 50
config.tracking_config.disappear_weight = -0.2

pprint(config.dict())


# In[10]:


track(
    foreground=detections,
    edges=edges,
    config=config,
    overwrite=True,
)


# In[11]:


tracks_df, graph = to_tracks_layer(config)
tracks_df = close_tracks_gaps(
    tracks_df=tracks_df,
    max_gap=2,
    max_radius=50,
    spatial_columns=["y", "x"],
)


# In[12]:


labels = tracks_to_zarr(
    config, tracks_df
)  # incase needed for napari or other CZI-specific applications
tracks_df.to_parquet(
    f"{results_output_dir}/{str(input_dir).split('MaxIP_')[1]}_tracks.parquet"
)
print(tracks_df["track_id"].nunique())
print(f"Found {tracks_df['track_id'].nunique()} unique tracks in the dataset.")
tracks_df.head()


# In[13]:


tracks_df.reset_index(drop=True, inplace=True)
cum_tracks_df = tracks_df.copy()
timepoints = tracks_df["t"].unique()

# zero out the track_df for plotting
cum_tracks_df = cum_tracks_df.loc[cum_tracks_df["t"] == -1]


# In[14]:


nuclei = nuclei * 4096
for frame_index, _ in enumerate(nuclei):
    tmp_df = tracks_df.loc[tracks_df["t"] == frame_index]
    cum_tracks_df = pd.concat([cum_tracks_df, tmp_df])
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 3, 1)
    # rescale the intensity of the raw image

    plt.imshow(nuclei[frame_index, :, :], cmap="gray")
    plt.title("Raw")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(detections[frame_index, :, :], cmap="gray")
    plt.title("Masks")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    sns.lineplot(data=cum_tracks_df, x="x", y="y", hue="track_id", legend=False)
    plt.imshow(detections[frame_index, :, :], cmap="gray", alpha=0.5)
    plt.title(f"Frame {frame_index}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{temporary_output_dir}/tracks_{frame_index}.png")
    plt.close()


# In[15]:


# load each image
files = [f for f in temporary_output_dir.glob("*.png")]
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
frames = [Image.open(f) for f in files]
fig_path = figures_output_dir / f"{str(input_dir).split('MaxIP_')[1]}_tracks.gif"
# plot the line of each track in matplotlib over a gif
# get the tracks
# save the frames as a gif
frames[0].save(fig_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


# In[16]:


# clean up tracking files
# remove temporary_output_dir
shutil.rmtree(temporary_output_dir)

track_db_path = pathlib.Path("data.db").resolve()
metadata_toml_path = pathlib.Path("metadata.toml").resolve()
if track_db_path.exists():
    track_db_path.unlink()
if metadata_toml_path.exists():
    metadata_toml_path.unlink()


# In[17]:


clear_gpu_memory()


# In[18]:


if in_notebook:
    viewer = napari.Viewer()
    viewer.window.resize(1200, 1200)

    viewer.add_image(nuclei, name="Raw", colormap="gray")

    viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph)
    viewer.add_labels(masks)

    viewer.layers["masks"].visible = False

    nbscreenshot(viewer)

