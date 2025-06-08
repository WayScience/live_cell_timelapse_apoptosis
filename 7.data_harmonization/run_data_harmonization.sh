#!/bin/bash

# Activate the conda environment

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to scripts
cd scripts/ || exit

# Run the scripts
# conda run -n cellprofiler_timelapse_env python 0.combine_CP_and_scDINO_features.py
conda run -n cellprofiler_timelapse_env python 1.normalize_combined_features.py
conda run -n cellprofiler_timelapse_env python 2.feature_select_combined_features.py
conda run -n cellprofiler_timelapse_env python 3.aggregation.py

# change the directory to the parent directory
cd .. || exit

echo "Data harmonization is done!"
