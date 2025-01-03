#!/bin/bash

# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/ || exit

echo "Starting IC processing"

# Run CellProfiler for IC processing
conda run -n cellprofiler_timelapse_env python 0.perform_ic.py # --test_data True
conda run -n cellprofiler_timelapse_env python 1.process_ic_teminal_data.py # --test_data

# return the directory that script was run from
cd ../ || exit

echo "IC processing complete"
