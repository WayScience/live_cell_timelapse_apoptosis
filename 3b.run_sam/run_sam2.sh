#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=6:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=sample-%j.out


# run SAM2
echo "Starting the CellProfiler analysis..."

# run SAM2
module load cuda
module load anaconda
module load mambaforge
conda init bash
mamba activate sam2_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/

python 1.run_sam2_microscopy.py
cd ../../

mamba deactivate
