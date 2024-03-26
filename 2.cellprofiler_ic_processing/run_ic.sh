#!/bin/bash

conda activate cellprofiler_timelapse_env

# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/

echo "Starting IC processing"

# Run CellProfiler for IC processing
python perform_ic.py

# return the directory that script was run from
cd ../

echo "IC processing complete"
