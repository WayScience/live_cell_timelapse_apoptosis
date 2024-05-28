#!/bin/bash
# This script processes the timelapse data

# activate the main conda environment
conda activate cellprofiler_timelapse_env

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

# move to the scripts directory
cd scripts

# run the python scripts in order (from convert+merge, annotate, normalize, feature select, and extract image features)
echo "Starting processing single cells"

echo "Converting and merging singles"
time python 0.merge_sc.py

echo "Annotating single cells"
time python 1.annotate_sc.py

echo "Normalizing single cells"
time python 2.normalize_sc.py

echo "Feature selecting single cells"
time python 3.feature_select_sc.py

# revert to the original directory
cd ..

# deactivate the conda environment
conda deactivate

# Complete
echo "Processing of single cells complete"
