#!/usr/bin/env python
# coding: utf-8

# This notebook solves the cell tracking issue by using [SAM2](https://github.com/facebookresearch/segment-anything-2/tree/main) instead of the functionality within CellProfiler.
# Here I use the pretrained model to segment the nuclei in the video.
# The output is a mask for each object in each frame and the x,y coordinates centers of each object in each frame.

# This is a notebook that needs perfect conditions to work.
# With a GeForce RTX 3090 TI, the 24GB of VRAM sometimes are not enough to process the videos.
#
# Hold your breath, pick a four-leaf clover, avoid black cats, cracks, and mirrors, and let's go!
#
# This notebook is converted to a script and ran from script to be compatible with HPC cluster.

# # Table of Contents for this Notebook
# #### 1. Imports
# #### 2. Import data
# #### 3. get the masks and centers
# #### 4. Track multiple objects in the video
# #### 5. Track the objects through frames
# #### 6. Visualize the tracking and output the data

# ## 1. Imports

# In[1]:


# top level imports
import gc  # garbage collector
import logging  # logging
import pathlib  # path handling
import shutil  # file handling
import subprocess  # subprocess handling
import sys  # system

import lancedb  # lancedb database
import matplotlib.pyplot as plt  # plotting
import numpy as np  # numerical python
import pandas as pd  # data handling
import pyarrow as pa  # pyarrow for parquet
import torch  # pytorch deep learning
from csbdeep.utils import Path, normalize  # dependecy for stardist
from PIL import Image  # image handling
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # sam2 image predictor
from skimage import io  # image handling
from skimage.measure import label, regionprops  # coordinate handling
from skimage.transform import resize  # image handling
from stardist.models import StarDist2D  # stardist
from stardist.plot import render_label  # stardist
from torchvision import models  # pytorch models

sys.path.append("../../utils/")
from SAM2_utils import (  # sam2 utils
    delete_recorded_memory_history,
    export_memory_snapshot,
    generate_random_coords,
    show_mask,
    show_points,
    start_record_memory_history,
    stop_record_memory_history,
)

# check cuda devices
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# ## 2. Import data

# ### Download the model(s)

# In[2]:


models_dict = {
    "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


# In[3]:


# Download the file using wget
# this is the model checkpoint for the SAM2 model
for file in models_dict.keys():
    model_path = pathlib.Path(file).resolve()
    new_model_path = pathlib.Path("../../data/models").resolve() / model_path.name
    # check if the model already exists
    if not new_model_path.exists():
        subprocess.run(["wget", models_dict[file]], check=True)
        new_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(model_path, new_model_path)
    else:
        print(f"Model {new_model_path} already exists. Skipping download.")


# In[4]:


# load in the model and the predictor
sam2_checkpoint = pathlib.Path("../../data/models/sam2_hiera_tiny.pt").resolve()
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# set the path to the videos

ordered_tiffs = pathlib.Path("../sam2_processing_dir/tiffs/").resolve()
converted_to_video_dir = pathlib.Path("../sam2_processing_dir/pngs/").resolve()
if converted_to_video_dir.exists():
    shutil.rmtree(converted_to_video_dir)

ordered_tiffs.mkdir(parents=True, exist_ok=True)
converted_to_video_dir.mkdir(parents=True, exist_ok=True)


# In[5]:


tiff_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_6hr_4ch_MaxIP_test_small"
).resolve(strict=True)
terminal_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small"
).resolve(strict=True)


# In[6]:


# create the database object
uri = pathlib.Path("../../data/objects_db").resolve()
db = lancedb.connect(uri)


# ### Get data formatted correctly

# In[7]:


# get the list of tiff files in the directory
tiff_files = list(tiff_dir.glob("*.tiff"))
tiff_files = tiff_files + list(terminal_dir.glob("*.tiff"))
tiff_file_names = [file.stem for file in tiff_files]
# files to df
tiff_df = pd.DataFrame({"file_name": tiff_file_names, "file_path": tiff_files})

# split the file_path column by _ but keep the original column
tiff_df["file_name"] = tiff_df["file_name"].astype(str)
tiff_df[["Well", "FOV", "Timepoint", "Z-slice", "Channel", "illum"]] = tiff_df[
    "file_name"
].str.split("_", expand=True)
tiff_df["Well_FOV"] = tiff_df["Well"] + "_" + tiff_df["FOV"]
# drop all channels except for the first one
tiff_df = tiff_df[tiff_df["Channel"] == "C01"]
tiff_df = tiff_df.drop(columns=["Channel", "illum"])
tiff_df["new_path"] = (
    str(ordered_tiffs)
    + "/"
    + tiff_df["Well_FOV"]
    + "/"
    + tiff_df["file_name"]
    + ".tiff"
)
tiff_df.reset_index(drop=True, inplace=True)
tiff_df.head()


# In[8]:


# copy the files to the new directory
# from file path to new path
for index, row in tiff_df.iterrows():
    new_path = pathlib.Path(row["new_path"])
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(row["file_path"], new_path)


# In[9]:


# get the list of directories in the ordered tiffs directory
ordered_tiff_dirs = list(ordered_tiffs.glob("*"))
ordered_tiff_dir_names = [dir for dir in ordered_tiff_dirs]
ordered_tiff_dir_names
for dir in ordered_tiff_dir_names:
    out_dir = converted_to_video_dir / dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for tiff_file in dir.glob("*.tiff"):
        jpeg_file = pathlib.Path(f"{out_dir}/{tiff_file.stem}.jpeg")

        if not jpeg_file.exists():
            try:
                with Image.open(tiff_file) as img:
                    # Convert the image to 8-bit per channel
                    img = img.convert("L")
                    img.save(jpeg_file)
            except Exception as e:
                print(f"Failed to convert {tiff_file}: {e}")


# In[10]:


# get list of dirs in the converted to video dir
converted_dirs = list(converted_to_video_dir.glob("*"))
converted_dir_names = [dir for dir in converted_dirs]
for dir in converted_dir_names:
    dir = sorted(dir.glob("*.jpeg"))
    for i in enumerate(dir):
        # rename the files to be in order
        i[1].rename(f"{dir[0].parent}/{str(i[0] + 1).zfill(3)}.jpeg")


# ### Donwsample each frame to fit the images on the GPU - overwrite the copies JPEGs

# In[11]:


# get files in the directory
converted_dirs_list = list(converted_to_video_dir.rglob("*"))
converted_dirs_list = [f for f in converted_dirs_list if f.is_file()]
# posix path to string
files = [str(f) for f in converted_dirs_list]


# In[12]:


# need to downscale to fit the model and images on the GPU
# note that this is an arbitrary number and can be changed
downscale_factor = 10
# sort the files by name
# downsample the image
for f in files:
    img = io.imread(f)
    # downsample the image
    downsampled_img = img[::downscale_factor, ::downscale_factor]
    # save the downsampled image in place of the original image
    io.imsave(f, downsampled_img)


# ## 3. Get initial masks and centers via StarDist

# ### Get the first frame of each video
# ### Set up a dict that holds the images path, the first frame_mask, and the first frame_centers

# In[13]:


# where one image set here is a single well and fov over all timepoints
image_set_dict = {
    "image_set_name": [],  # e.g. well_fov
    "image_set_path": [],  # path to the directory
    "image_set_first_frame": [],  # path to the first frame
    "image_x_y_coords": [],  # list of x,y coordinates
    "image_labels": [],  # list of labels for the x,y coordinates
}

# get the list of directories in the ordered tiffs directory
dirs = list(converted_to_video_dir.glob("*"))
dirs = [dir for dir in dirs if dir.is_dir()]
for dir in dirs:
    # get the files in the directory
    files = sorted(dir.glob("*.jpeg"))
    image_set_dict["image_set_name"].append(dir.name)
    image_set_dict["image_set_path"].append(str(dir))
    image_set_dict["image_set_first_frame"].append(files[0])


# ### Plot the segementation
# Plot the following:
# - the original image
# - the segmentation
# - the x,y centers of the segmentation
# - the extracted masks

# In[14]:


model = StarDist2D.from_pretrained("2D_versatile_fluo")

# choose to visualize the results or not
# best for troubleshooting or exploring the model
visualize = False

# loop through each image set and predict the instances
for i in range(len(image_set_dict["image_set_name"])):
    print(
        f"{image_set_dict['image_set_name'][i]}: {image_set_dict['image_set_first_frame'][i]}"
    )
    img = io.imread(image_set_dict["image_set_first_frame"][i])
    labels, _ = model.predict_instances(normalize(img))
    # convert the labels into position coordinates
    regions = regionprops(label(labels))
    coords = np.array([r.centroid for r in regions])
    coords = coords[:, [1, 0]]

    if visualize:
        # plot the points and the masks and the image side by side by side
        fig, ax = plt.subplots(1, 4, figsize=(30, 15))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(render_label(labels, img=img))
        ax[1].set_title("Masks")
        ax[2].imshow(img, cmap="gray")
        ax[2].scatter(
            coords[:, 1],
            coords[:, 0],
            color="red",
            marker="*",
            s=100,
            edgecolor="white",
            linewidth=1.25,
        )
        ax[2].set_title("Points")

        ax[3].invert_yaxis()
        # make the aspect ratio equal
        ax[3].set_aspect("equal")
        show_points(coords, np.ones(len(coords)), ax[3])
    labels = np.ones(coords.shape[0], dtype=np.int32)
    image_set_dict["image_x_y_coords"].append(coords)
    image_set_dict["image_labels"].append(labels)

# remove star dist model from memory
del model
# remove all stardist gpu memory
torch.cuda.empty_cache()


# ## 4. Track multiple objects in the video

# ### Begin GPU Profiling

# In[15]:


# Start recording memory snapshot history
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
# delete any prior memory profiling data
delete_recorded_memory_history(
    logger=logger, save_dir=pathlib.Path("../memory_snapshots/").resolve()
)

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
start_record_memory_history(
    logger=logger, max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
)


# In[16]:


# clear the memory
torch.cuda.empty_cache()
gc.collect()


# In[17]:


stored_video_segments = {}


# In[18]:


# loop through each image set and predict the instances
for i in range(len(image_set_dict["image_set_name"])):
    print(
        f"{image_set_dict['image_set_name'][i]}: {image_set_dict['image_set_first_frame'][i]}"
    )
    frame_names = sorted(list(Path(image_set_dict["image_set_path"][i]).glob("*.jpeg")))
    img = io.imread(frame_names[0])
    h, w = img.shape
    print(h, w)
    # initialize the state
    inference_state = predictor.init_state(
        video_path=str(image_set_dict["image_set_path"][i]),
        offload_video_to_cpu=True,  # set to True if the video is too large to fit in GPU memory
        offload_state_to_cpu=True,  # set to True if the state is too large to fit in GPU memory
    )
    predictor.reset_state(inference_state)
    prompts = {}
    ann_frame_idx = 0
    ann_obj_idx = 1
    samples = 1
    negative_sampling = (
        False  # set True to generate negative samples for better training
    )
    # loop through the points and add them to the state and get the masks
    for _point, _label in zip(
        image_set_dict["image_x_y_coords"][i], image_set_dict["image_labels"][i]
    ):
        _label = np.array([_label], dtype=np.int32)
        _point = np.array([_point], dtype=np.float32)

        if negative_sampling:
            random_points, random_labels = generate_random_coords(
                img=img, coords=_point, samples=samples
            )
            _point = np.concatenate([_point, random_points], axis=0)
            _label = np.concatenate([_label, random_labels], axis=0)
        # add the points to the state
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_idx,
            points=_point,
            labels=_label,
        )
        # save the prompts
        prompts[ann_obj_idx] = {
            "points": _point,
            "labels": _label,
            "out_obj_ids": out_obj_ids[0],
            "out_mask_logits": out_mask_logits[0].detach().cpu().numpy(),
        }
        # increment the object index for this frame
        ann_obj_idx += 1

    del prompts
    del samples
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(range(1, ann_obj_idx))
        }
    stored_video_segments[image_set_dict["image_set_name"][i]] = video_segments

    # clear the memory
    del inference_state

    del out_mask_logits
    del out_obj_ids
    del out_frame_idx
    torch.cuda.empty_cache()
    gc.collect()


# ### stop GPU profiling

# In[ ]:


# save the memory snapshot to a file
export_memory_snapshot(
    logger=logger, save_dir=pathlib.Path("../memory_snapshots/").resolve()
)
stop_record_memory_history(logger=logger)


# In[ ]:


# remove previous runs generated files
# each of these directories will be created if they do not exist
# the new files will be saved in these directories

# for masks
masks_dir = pathlib.Path("../sam2_processing_dir/masks").resolve()
if masks_dir.exists():
    shutil.rmtree(masks_dir)
masks_dir.mkdir(exist_ok=True, parents=True)

# for gifs
gifs_dir = pathlib.Path("../sam2_processing_dir/gifs").resolve()
if gifs_dir.exists():
    shutil.rmtree(gifs_dir)
gifs_dir.mkdir(exist_ok=True, parents=True)

# for combined masks and tiffs
combined_dir = pathlib.Path("../sam2_processing_dir/CP_input").resolve()
if combined_dir.exists():
    shutil.rmtree(combined_dir)
combined_dir.mkdir(exist_ok=True, parents=True)


# In[ ]:


output_dict = {
    "image_set_name": [],
    "frame": [],
    "object_id": [],
    "x": [],
    "y": [],
    "mask_path": [],
    "mask_file_name": [],
}


# In[ ]:


# loop through each image set and save the predicted masks as images
for i in range(len(image_set_dict["image_set_name"])):
    print(
        f"{image_set_dict['image_set_name'][i]}: {image_set_dict['image_set_first_frame'][i]}"
    )
    frame_names = sorted(list(Path(image_set_dict["image_set_path"][i]).glob("*.jpeg")))
    img = io.imread(frame_names[0])
    h, w = img.shape
    upscale_h = h * downscale_factor
    upscale_w = w * downscale_factor
    print(h, w, "upscaled", upscale_h, upscale_w)
    # add all of the frames together for a rendered gif
    # create a list of all the frames
    frames = []

    video_segments = stored_video_segments[image_set_dict["image_set_name"][i]]
    for out_frame_idx in range(0, len(frame_names), 1):
        # create a figure
        # set the frame path and make the directory if it doesn't exist
        # create a frame image
        frame_image = np.zeros((h, w), dtype=np.uint8)
        # loop through the objects in the frame
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # add the mask to the frame image
            frame_image += (out_mask[0] * 255).astype(np.uint8)
            out_mask = np.array(out_mask[0], dtype=np.float32)
            # convert the outmask to an image
            regions = regionprops(label(out_mask))
            for region in regions:
                y, x = region.centroid
                # scale the x and y coordinates back to the original size
                x = x * downscale_factor
                y = y * downscale_factor
                output_dict["frame"].append(out_frame_idx)
                output_dict["object_id"].append(out_obj_id)
                output_dict["x"].append(x)
                output_dict["y"].append(y)
                output_dict["mask_file_name"].append(f"{out_frame_idx}.png")
                output_dict["image_set_name"].append(
                    image_set_dict["image_set_name"][i]
                )
                output_dict["mask_path"].append(masks_dir)

        # save the frame image
        # scale the image upscale back to the original size
        frame_image = Image.fromarray(frame_image)
        frame_image = frame_image.resize((upscale_w, upscale_h), Image.NEAREST)

        # convert the frame image to ints
        frame_image_path = f"{masks_dir}/{image_set_dict['image_set_name'][i]}_T{str(out_frame_idx + 1).zfill(4)}_Z0001_mask.png"
        frame_image.save(frame_image_path)

        # add title to the subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # show the image
        ax.imshow(frame_image, cmap="gray")
        ax.set_title(f"Frame {out_frame_idx}")
        # save the figure to a file
        fig.savefig(f"tmp_{out_frame_idx}.png")
        # close the figure
        plt.close(fig)
        # open the image
        img = Image.open(f"tmp_{out_frame_idx}.png")
        # append the image to the frames
        frames.append(img)

    fig_path = pathlib.Path(
        f"{gifs_dir}/{image_set_dict['image_set_name'][i]}_out.gif"
    ).resolve()
    # save the frames as a gif
    frames[0].save(
        fig_path, save_all=True, append_images=frames[1:], duration=10, loop=0
    )

    # get all files that have tmp in the name
    tmp_files = list(Path(".").glob("tmp*.png"))
    # delete all the tmp files
    [f.unlink() for f in tmp_files]


# In[ ]:


file_paths_df = pd.DataFrame(output_dict)
# add the mask file path
file_paths_df["mask_file_path"] = (
    file_paths_df["mask_path"].astype(str)
    + "/"
    + file_paths_df["mask_file_name"].astype(str)
)
# type cast the columns
file_paths_df["image_set_name"] = file_paths_df["image_set_name"].astype(str)
file_paths_df["frame"] = file_paths_df["frame"].astype(np.int32)
file_paths_df["object_id"] = file_paths_df["object_id"].astype(np.int32)
file_paths_df["x"] = file_paths_df["x"].astype(np.float32)
file_paths_df["y"] = file_paths_df["y"].astype(np.float32)
file_paths_df["mask_path"] = file_paths_df["mask_path"].astype(str)
file_paths_df["mask_file_name"] = file_paths_df["mask_file_name"].astype(str)
file_paths_df["mask_file_path"] = file_paths_df["mask_file_path"].astype(str)
# add to the db
# set up schema
schema = pa.schema(
    [
        pa.field("image_set_name", pa.string()),
        pa.field("frame", pa.int32()),
        pa.field("object_id", pa.int32()),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("mask_path", pa.string()),
        pa.field("mask_file_name", pa.string()),
        pa.field("mask_file_path", pa.string()),
    ]
)

# create the table
tbl = db.create_table("1.masked_images", schema=schema, mode="overwrite")
# write the data to the table
tbl.add(file_paths_df)


# In[ ]:


# read the data from the table and check the first few rows
tbl.to_pandas().head()
