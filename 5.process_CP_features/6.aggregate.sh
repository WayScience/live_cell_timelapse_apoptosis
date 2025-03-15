#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=5:00:00
#SBATCH --output=ic_combine-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 6.aggregation.py

cd ../ || exit

conda deactivate

echo "All merging sc jobs submitted."
