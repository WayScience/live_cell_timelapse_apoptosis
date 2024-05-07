# Live Cell Timelapse Apoptosis Analysis

This repository contains the code to analyze live cell timelapse microscopy data of apoptosis in HeLa cells.
The goal of this analysis repository is to build a pipeline and framework to analyze live cell timelapse microscopy data modes.
Specifically, multi-channel fluorescence microscopy data.
We will do use by extracting morphology features from images using [CellProfiler](https://cellprofiler.org/) and using the coordinate information to extract single cell representation using a self supervised learning approach.
The self supervised learning approach is implemented by using [scDINO](https://github.com/JacobHanimann/scDINO).

The sample data and upstream analysis of such data, including image analysis with CellProfiler and image-profiling with CytoTable and pycytominer, can be found in the sibling repository [live_cell_timelapse_apoptosis](https://github.com/WayScience/live_cell_timelapse_apoptosis).

## Use
For each of the modules there are specific conda environments that are used to run the code.
The conda environments can be found in the [environments](environments) directory.
For instructions on how to create the conda environments, please refer to the [README](environments/README.md) in the environments(environments) directory.
