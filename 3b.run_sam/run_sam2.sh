#!/bin/bash

# run SAM2


jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/
mamba activate sam2_env
python 1.run_sam2_microscopy.py
cd ../../

mamba deactivate
