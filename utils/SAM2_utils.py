"""
These helper functions are specifically for running SAM2.
For object detection and segmentation, over time.
"""

import logging
import pathlib
import socket
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


### Visualizing the output masks and points
def show_mask(
    mask: np.ndarray,
    ax: plt.Axes,
    obj_id: int | None = None,
    random_color: bool = False,
) -> None:
    """
    Show the mask on the ax.
    Modified from: https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb

    Parameters
    ----------
    mask : np.ndarray
        The mask to be shown.
    ax : plt.Axes
        The ax to show the mask.
    obj_id : int | None, optional
        The object id. If None, the color will be randomly generated.
    random_color : bool, optional
        If True, the color will be randomly generated. If False, the color will be generated based on the object id.

    Returns
    -------
    None
    """

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(
    coords: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes,
    marker_size: int = 100,
) -> None:
    """
    Show the points on the ax.
    Modified from: https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the points.
    labels : np.ndarray
        The labels of the points.
    ax : plt.Axes
        The ax to show the points.
    marker_size : int, optional
        The size of the marker.

    Returns
    -------
    None
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="blue",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )  #
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def generate_random_coords(
    img: np.array,
    coords: np.array,
    samples: int,
    show_masks: bool = False,
    search_space_pixels: int = 25,
) -> Tuple[np.array, np.array]:
    """
    Generate random coordinates that are not in the mask for negative samples

    Parameters
    ----------
    img : np.array
        The image to generate the random coordinates
    coords : np.array
        Position coordinates of the masks
    samples : int
        The number of samples to generate
    show_masks : bool
        Whether to show the masks or not
    search_space_pixels : int
        The number of pixels to search for the negative samples away from the mask

    Returns
    -------
    Tuple[np.array, np.array]
        Random coordinates that are not in the mask and the labels for the coordinates
    """
    seed = 0
    # get the height and width of the image
    h, w = img.shape
    # generate random coords
    rand_coords = np.column_stack(
        [np.random.randint(0, h, samples), np.random.randint(0, w, samples)]
    )

    valid_coords = []
    # check if the coords are within the specified pixels of the existing coords
    for coord in rand_coords:
        # calculate the distance to the existing coords
        # check if the distance is less than the pixels specified
        if not np.any(np.linalg.norm(coord - coords, axis=1) < search_space_pixels):
            try:
                if not np.any(
                    np.linalg.norm(coord - np.array(valid_coords), axis=1)
                    < search_space_pixels
                ):
                    valid_coords.append(coord)
            except:
                valid_coords.append(coord)

    valid_coords = np.array(valid_coords)
    valid_labels = np.zeros(valid_coords.shape[0])

    if show_masks:
        # plot the points
        # plot size
        plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots()
        # ax.imshow(img, cmap="gray")
        # plot a square where each point is
        for coord in coords:
            ax.plot(coord[1], coord[0], "o", color="red")
        for coord in valid_coords:
            ax.plot(coord[1], coord[0], "o", color="cyan")
        plt.show()

    # must return at least 1 valid coord
    if valid_coords.shape[0] == 0:
        valid_coords, valid_labels = generate_random_coords(
            img=img, coords=coords, samples=samples
        )

    return valid_coords, valid_labels


### Memory profiling for GPU
def start_record_memory_history(logger: logging.Logger, max_entries: int) -> None:
    """
    Start recording memory history.

    Parameters
    ----------
    logger : logging.Logger
        The logger to log the information
    max_entries : int
        The maximum number of entries to record

    Returns
    -------
    None
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    return None


def stop_record_memory_history(logger: logging.Logger) -> None:
    """
    Stop recording memory history.

    Parameters
    ----------
    logger : logging.Logger
        The logger to log the information

    Returns
    -------
    None
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)
    return None


def export_memory_snapshot(
    logger: logging.Logger, save_dir: pathlib.Path, time_format: str = "%b_%d_%H_%M_%S"
) -> None:
    """
    Export memory snapshot.

    Parameters
    ----------
    logger : logging.Logger
        The logger to log the information
    save_dir : pathlib.Path
        The directory to save the memory snapshot
    time_format : str, optional
        The time format for the file name

    Returns
    -------
    None
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(time_format)
    file_prefix = f"{host_name}_{timestamp}"
    full_file_path = pathlib.Path(save_dir / f"{file_prefix}.pickle").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Saving snapshot to local file: {full_file_path}")
        torch.cuda.memory._dump_snapshot(full_file_path)
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def delete_recorded_memory_history(
    logger: logging.Logger, save_dir: pathlib.Path
) -> None:
    """
    Delete recorded memory history.

    Parameters
    ----------
    logger : logging.Logger
        The logger to log the information
    save_dir : pathlib.Path
        The directory to save the memory history

    Returns
    -------
    None
    """
    file_path = pathlib.Path(save_dir).resolve()
    files = list(file_path.glob("*.pickle"))
    try:
        [file.unlink() for file in files]
        logger.info(f"Deleted memory snapshot file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to delete memory snapshot file: {file_path} {e}")
        return
