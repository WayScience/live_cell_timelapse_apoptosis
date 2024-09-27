#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=6:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=alpine_std_out_std_err-%j.out


# run SAM2
echo "Starting SAM2 pipe for object detection..."

# run SAM2
module load cuda
module load anaconda
module load mambaforge
conda init bash
mamba activate sam2_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/

# run the pipelines
python 0.create_db_for_pipe.py
python 1.run_sam2_microscopy.py
cd ../../

mamba deactivate
