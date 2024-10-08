#!/bin/bash

# run SAM2
echo "Starting SAM2 pipe for object detection..."

# run SAM2

mamba activate sam2_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/

# run the pipelines
python 0.create_db_for_pipe.py
python 1.run_sam2_microscopy.py
cd ../../

mamba deactivate
