#!/bin/bash
# This script processes the timelapse data
module purge

module load anaconda

# activate the main conda environment
conda activate interstellar_data

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

# run the python scripts in order (from convert+merge, annotate, normalize, feature select, and extract image features)
echo "Starting processing single cells"

echo "Converting and merging singles"
time python scripts/0.merge_sc.py

echo "Annotating single cells"
time python scripts/1.annotate_sc.py

echo "Normalizing single cells"
time python scripts/2.normalize_sc.py

echo "Feature selecting single cells"
time python scripts/3.feature_select_sc.py

echo "Processing of single cells complete"
