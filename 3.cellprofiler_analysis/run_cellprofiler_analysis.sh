#!/bin/bash

# This script is used to run CellProfiler analysis on the timelapse images.


# activate the conda environment
conda activate cellprofiler_timelapse_env

# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/

# run the python script
time python run_cellprofiler_analysis.py

# deactivate the conda environment
conda deactivate

# change the directory back to the orginal directory
cd ../

# End of the script
echo "CellProfiler analysis is completed."
