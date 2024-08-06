#!/bin/bash

# This script creates a conda environment for SAM2 and installs the required dependencies.
# SAM2 is installed from source.

mamba env remove -n sam2_env -y
mamba env create -f sam2.yaml
module purge
mamba activate sam2_env

git clone git@github.com:facebookresearch/segment-anything-2.git
cd segment-anything-2
git checkout 57bc94b7391e47e5968004a0698f8bf793a544d1

module load cuda/12.1
pip install --no-build-isolation -e .

cd ../

mamba deactivate

echo "Environment setup complete, SAM2 is ready to use."
