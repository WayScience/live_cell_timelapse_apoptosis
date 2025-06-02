#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --partition=amem
#SBATCH --qos=mem
#SBATCH --account=amc-general
#SBATCH --time=5:00:00
#SBATCH --output=Feature_selection-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 5.feature_select_sc.py

cd ../ || exit

conda deactivate

echo "Feature selection complete."
