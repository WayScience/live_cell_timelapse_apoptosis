#!/bin/bash

# this script runs the file and pathing pre-processing

# change directory to the scripts folder
cd scripts/

# activate the conda environment
conda activate timelapse_env

# run the pre-processing scripts
python 0.fix_pathing.py
python 1.generate_platemap.py

# revert back to the main directory
cd ../

# deactivate the conda environment
conda deactivate

echo "Pre-processing complete"
