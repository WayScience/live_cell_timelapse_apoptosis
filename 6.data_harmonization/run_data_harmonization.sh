#!/bin/bash

# Activate the conda environment

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to scripts
cd scripts/

# Run the scripts
conda run -n cellprofiler_timelapse_env python 0.combine_CP_and_scDINO_features.py
conda run -n cellprofiler_timelapse_env python 1.normalize_combined_features.py
conda run -n cellprofiler_timelapse_env python 2.feaure_select_combined_features.py

# change the directory to the parent directory
cd ..

echo "Data harmonization is done!"
