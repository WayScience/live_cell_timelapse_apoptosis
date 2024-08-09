#!/bin/bash

# This script is used to run CellProfiler analysis on the timelapse images.

# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/

# run the python script
conda run -n cellprofiler_timelapse_env python run_cellprofiler_analysis.py

# change the directory back to the orginal directory
cd ../

# End of the script
echo "CellProfiler analysis is completed."
