#!/bin/bash

# create conda environments needed for the project

# loop through all environment .yaml files in this directory
for file in $(ls -1 *.yaml); do
    # create conda environment from .yaml file
    mamba env create -f $file
done


# set up plugins for cellprofiler
/home/lippincm/miniforge3/envs/cellprofiler_timelapse_env/bin/cellprofiler
