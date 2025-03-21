#!/bin/bash

# activate the correct env
conda activate scDINO_env

# convert notebooks into scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change to the correct directory
cd scripts || exit

# run the script
time python 0.pre-process_images.py
time python 1.calculate_mean_std_per_channel.py

# revert to the original directory
cd .. || exit

# deactivate the env
conda deactivate

# Complete
echo "Image pre-processing complete"
