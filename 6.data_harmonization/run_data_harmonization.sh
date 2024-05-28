#!/bin/bash

# Activate the conda environment

conda activate cellprofiler_timelapse_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to scripts
cd scripts/

# Run the scripts
time python 0.combine_CP_and_scDINO_features.py
time python 1.normalize_combined_features.py
time python 2.feaure_select_combined_features.py

# change the directory to the parent directory
cd ..

# Deactivate the conda environment
conda deactivate

echo "Data harmonization is done!"
